from pathlib import Path

def main():
    p = Path('footage')

    if not p.exists():
        print("error: footage folder not found")

    with open('videos.txt', 'w') as f:
        for file in p.glob("**/*.MOV"):
            if not file.is_dir():
                f.write(str(file) + "\n")

if __name__ == '__main__':
    main()