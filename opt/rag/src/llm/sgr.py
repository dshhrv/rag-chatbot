import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import requests
import re
import time
import requests
import os
from collections import defaultdict

from src.retrieval.bm25 import bm25_search, load_index, INDEX_PATH
from src.retrieval.dense import dense_search, COLLECTION_NAME
from src.retrieval.rrf import rrf_fuse
from src.retrieval.glossary import detect_terms, get_definitions
from scripts.translate import en2ru
from src.retrieval.retrieve import retrieve_top

from src.llm.promts import PROMT

