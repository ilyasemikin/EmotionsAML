import torch

if __name__ == "__main__":
    from src.modules.predict import predict    
    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--input', default=None, required=True, type=str)
    args = parser.parse_args()

    result = predict(args.input)
    for item in result:
        print(item[0], item[1])
