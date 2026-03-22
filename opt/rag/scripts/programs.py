import json
import requests
from bs4 import BeautifulSoup
import re
from hashlib import sha1


CLAUSE_RE = re.compile(r"^\s*(?P<id>\d{2}\.\d{2}\.\d{2})\s(?P<name>.+?)\s*$")


def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_id(*parts, n=24):
    def norm(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " > ".join(norm(i) for i in x if i not in (None, ""))
        return str(x)
    payload = "|".join(norm(p) for p in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:n]

def programms_parser(OUT_PATH, URL, level, type="w"):
    html = requests.get(URL).text
    soup = BeautifulSoup(html, 'lxml')
    cards = soup.select("a.e-card.e-cards__item")
    items = []

    for card in cards:
        items.append({
            "url": card.get("href"),
            "tags": card.select_one("ul.e-tags li.e-tag").get_text(" ", strip=True),
            "category": card.select_one("div.e-card__category").get_text(" ", strip=True),
            "title": card.select_one("h3.e-card__title .e-card__title-inner").get_text(" ", strip=True),
            "city": card.select_one('li[title="Кампус"]').get_text(" ", strip=True),
        })
    with open (OUT_PATH, type, encoding="utf-8") as f:
        for item in items:
            m = CLAUSE_RE.match(item["category"])
            if m:
                program_id = m.group("id")
                name = m.group("name")
                category_full = item["category"]
                title = item["title"]
                url = item["url"]
                tags = item["tags"]
                city = item["city"]
                program_name_full = program_id + " " + name + " " + title
                id = make_id(program_id, title, name, url, level, city)
                meta = {
                    "url": url,
                    "level": level,
                    "city": city,
                    "tags": tags,
                    "program_id": program_id,
                    "name": name,
                    "title": title,
                    "category_full": category_full,
                }
                obj = {
                    "id": id,
                    "text": program_name_full,
                    "meta": meta
                }
                dump_line(f, obj)



if __name__ == "__main__":
    url_bak = "https://www.hse.ru/n/education/bachelor?pageSize=all"
    url_mag = "https://www.hse.ru/n/education/magister?pageSize=all"
    OUT_PATH = "programs.jsonl"
    programms_parser(
        OUT_PATH=OUT_PATH,
        URL=url_bak,
        level="bachelor",
        type="w"
    )
    programms_parser(
        OUT_PATH=OUT_PATH,
        URL=url_mag,
        level="master",
        type="a"
    )