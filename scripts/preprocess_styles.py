#!/usr/bin/env python3
import sys
sys.path.append('.')
from models.clip_retrieval import CLIPStyleRetriever
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style-dir', default='data/styles')
    parser.add_argument('--labels', default='data/style_labels.json')
    parser.add_argument('--output', default='checkpoints/style_features.pt')
    args = parser.parse_args()
    
    retriever = CLIPStyleRetriever(args.style_dir)
    retriever.preprocess_style_library(args.labels, args.output)
    print("\n✅ 预处理完成!")

if __name__ == '__main__':
    main()