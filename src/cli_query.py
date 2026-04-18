# src/cli_query.py
from rag import answer_with_rag

if __name__ == "__main__":
    while True:
        q = input("Spørsmål (blank for å avslutte): ")
        if not q.strip():
            break
        ans = answer_with_rag(q)
        print("\nSvar:\n", ans, "\n", "-"*60)
