import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "opt" / "rag"))

from src.retrieval.graph import GRAPH_PATH, build_graph, graph_retrieve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    args = parser.parse_args()
    if args.query is not None:
        for chunk in graph_retrieve(args.query):
            print(chunk["id"], chunk["text"])
    else:
        graph = build_graph()
        print(f"Saved {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations: {GRAPH_PATH}")
