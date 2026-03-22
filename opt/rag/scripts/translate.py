from transformers import MarianTokenizer, MarianMTModel
import torch
import json

MODEL_NAME_TRANSLATE = "Helsinki-NLP/opus-mt-en-ru"



def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@torch.no_grad()
def translate_en2ru(text, tok, model, device="cpu", max_new_tokens=128):
    batch = tok([text], return_tensors="pt", truncation=True).to(device)
    out_ids = model.generate(**batch, max_new_tokens=max_new_tokens)
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0]


# def translate_en2ru_golden(in_path, out_path, tok, model, device="cpu", max_new_tokens=128):
#     with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
#         for line in fin:
#             obj = json.loads(line)
#             if obj.get("lang") != "en":
#                 continue
#             text_ru = translate_en2ru(obj["text"], tok, model, device=device, max_new_tokens=max_new_tokens)
#             dump_line(
#                 fout,
#                 {
#                     "id": obj["id"],
#                     "lang": "ru",
#                     "text": text_ru,
#                     "rel": obj["rel"],
#                 },
#             )



def en2ru(query, tok, model, device="cpu", max_new_tokens=128):
    return translate_en2ru(query, tok=tok, model=model,
                           device=device, max_new_tokens=max_new_tokens)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--in", dest="in_path", type=str)
    parser.add_argument("--out", dest="out_path", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tok = MarianTokenizer.from_pretrained(MODEL_NAME_TRANSLATE)
    model = MarianMTModel.from_pretrained(MODEL_NAME_TRANSLATE).to(args.device)
    model.eval()

    if args.text is not None:
        print(en2ru(args.text, tok, model, device=args.device, max_new_tokens=args.max_new_tokens))
    else:
        translate_en2ru_golden(
            in_path=args.in_path,
            out_path=args.out_path,
            tok=tok,
            model=model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
