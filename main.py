import re
import numpy as np

def tekst(dokument: str) -> list[str]:
    return re.sub(r'[^\w\s]', '', dokument).lower().split(' ')

def pobierz_unikalne_slowa(*listy):
    slowa = set()
    for arr in listy:
        for slowo in arr:
            slowa.add(slowo)

    return slowa

def oblicz_c_q(parsowane_dokumenty: list[list[str]], zapytanie: list[str]) -> tuple[np.array, np.array]:
    wszystkie_slowa = pobierz_unikalne_slowa(*parsowane_dokumenty, zapytanie)
    C = np.zeros((len(wszystkie_slowa), len(parsowane_dokumenty)))
    q = np.zeros(len(wszystkie_slowa))
    for i, slowo in enumerate(wszystkie_slowa):
        if slowo in zapytanie:
            q[i] = 1
        for j, dokument in enumerate(parsowane_dokumenty):
            if slowo in dokument:
                C[i, j] = 1

    return C, q

def oblicz_cosinus(C_k: np.array, q_k: np.array) -> list:
    wynik = []
    for dokument in C_k.T:
        istotnosc = np.dot(dokument, q_k) / (np.linalg.norm(dokument) * np.linalg.norm(q_k))
        wynik.append(istotnosc)

    return wynik

def oblicz_istotnosc(C: np.array, q: np.array, k: int) -> np.array:
    U, s, Vt = np.linalg.svd(C)
    sk = np.take(s, range(k), axis=0)
    Sk = np.diag(sk)
    Vkt = np.take(Vt, range(k), axis=0)
    Ck = Sk.dot(Vkt)
    Ukt = np.take(U.T, range(k), axis=0)
    Sk1 = np.linalg.inv(Sk)
    q_k = Sk1.dot(Ukt).dot(q)

    return oblicz_cosinus(Ck, q_k)

def main():
    parsowane_dokumenty = []
    liczba_dokumentow = int(input())
    for _ in range(liczba_dokumentow):
        parsowane_dokumenty.append(tekst(input()))
    zapytanie = tekst(input())
    liczba_redukcji = int(input())
    C, q = oblicz_c_q(parsowane_dokumenty, zapytanie)
    istotnosc = oblicz_istotnosc(C, q, liczba_redukcji)
    sformatowana_istotnosc = list(map(lambda x: round(float(x), 2), istotnosc))
    print(sformatowana_istotnosc)

if __name__ == '__main__':
    main()