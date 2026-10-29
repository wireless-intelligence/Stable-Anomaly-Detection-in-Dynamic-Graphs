import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


URLS = {
    "BlogCatalog": "https://github.com/kaize0409/GCN_AnomalyDetection/raw/master/data/BlogCatalog.mat",
    "Flickr": "https://github.com/kaize0409/GCN_AnomalyDetection/raw/master/data/Flickr.mat",
    "ACM": "https://github.com/kaize0409/GCN_AnomalyDetection/raw/master/data/ACM.mat",
}


CITATION_FILES = {
    "Cora": ["https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.{}"],
    "Citeseer": ["https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.{}"],
    "Pubmed": ["https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.{}"],
}

CITATION_SUFFIXES = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
ENRON_URL = "https://snap.stanford.edu/data/email-Enron.txt.gz"
CTU13_URL = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"


CHECKS = {
    "Amazon": [
        "Amazon/Amazon.mat",
        "Amazon.mat",
    ],
    "YelpChi": [
        "yelp/YelpChi.mat",
        "YelpChi/YelpChi.mat",
        "YelpChi.mat",
    ],
    "Enron": [
        "Enron/Enron.mat",
        "Enron/email-Enron.txt",
    ],
    "CTU-13": [
        "CTU-13/CTU-13.mat",
        "CTU-13/ctu13.csv",
    ],
    "BlogCatalog": [
        "BlogCatalog/BlogCatalog.mat",
        "BlogCatalog/BlogCatalog.edgelist",
        "BlogCatalog.edgelist",
    ],
    "Flickr": [
        "Flickr/Flickr.mat",
    ],
    "ACM": [
        "ACM/ACM.mat",
    ],
    "Cora": [
        "ind.cora.x",
        "Cora/Cora/final_cora_real_feature.pkl",
        "cora/cora.cites",
    ],
    "Citeseer": [
        "ind.citeseer.x",
        "Citeseer/citeseer/citeseer.cites",
    ],
    "Pubmed": [
        "ind.pubmed.x",
        "Pubmed/final_pub_real_feature.pkl",
    ],
    "Arxiv": [
        "Arxiv/Arxiv/my_data/L1.pkl",
    ],
}


def _download(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  saved: {dest}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def download_mat(name: str, data_dir: str) -> bool:
    if name not in URLS:
        return False
    return _download(URLS[name], os.path.join(data_dir, name, f"{name}.mat"))


def download_citation(name: str, data_dir: str) -> bool:
    if name not in CITATION_FILES:
        return False
    ok = True
    name_low = name.lower()
    for url_tmpl in CITATION_FILES[name]:
        for suf in CITATION_SUFFIXES:
            url = url_tmpl.format(suf)
            dest = os.path.join(data_dir, f"ind.{name_low}.{suf}")
            if not _download(url, dest):
                ok = False
    return ok


def download_enron(data_dir: str) -> bool:
    import gzip, shutil
    os.makedirs(os.path.join(data_dir, "Enron"), exist_ok=True)
    gz = os.path.join(data_dir, "Enron", "email-Enron.txt.gz")
    if not _download(ENRON_URL, gz):
        return False
    txt = os.path.join(data_dir, "Enron", "email-Enron.txt")
    if not os.path.exists(txt):
        with gzip.open(gz, "rb") as fin, open(txt, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return True


def download_ctu13(data_dir: str) -> bool:
    target = os.path.join(data_dir, "CTU-13", "CTU-13-Dataset.tar.bz2")
    print("CTU-13 is large (~1.9 GB). After download, extract and place .binetflow files inside data/CTU-13/")
    return _download(CTU13_URL, target)


def inventory(data_dir: str):
    p, m = [], []
    for name, paths in CHECKS.items():
        ok = any(os.path.exists(os.path.join(data_dir, rel)) for rel in paths)
        if ok:
            p.append(name)
        else:
            m.append((name, [os.path.join(data_dir, rel) for rel in paths]))
    return p, m


def parse_args():
    pp = argparse.ArgumentParser()
    pp.add_argument("--dataset", type=str, default="all")
    pp.add_argument("--data_dir", type=str, default="./data")
    pp.add_argument("--inventory", action="store_true")
    return pp.parse_args()


def main():
    args = parse_args()
    if args.inventory:
        p, m = inventory(args.data_dir)
        print("Available datasets:")
        for n in p:
            print(f"  [OK]      {n}")
        print("Missing datasets:")
        for n, paths in m:
            print(f"  [MISSING] {n:<12} expected at one of: {paths[0]}")
        print()
        print("To download: python scripts/download_datasets.py --dataset <name>")
        return

    if args.dataset == "all":
        targets = list(URLS.keys()) + list(CITATION_FILES.keys()) + ["Enron", "CTU-13"]
    else:
        targets = [args.dataset]
    for t in targets:
        print(f"== {t} ==")
        if t in URLS:
            download_mat(t, args.data_dir)
        elif t in CITATION_FILES:
            download_citation(t, args.data_dir)
        elif t == "Enron":
            download_enron(args.data_dir)
        elif t == "CTU-13":
            download_ctu13(args.data_dir)
        else:
            print(f"  No automatic downloader for {t}; place manually under {args.data_dir}/{t}/")


if __name__ == "__main__":
    main()
