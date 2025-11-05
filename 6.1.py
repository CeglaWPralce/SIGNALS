import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definicja funkcji splotu ---
def my_convolution_sum(x, h):
    """
    Implementuje sumę splotu y[n] = x[n] * h[n] dla sygnałów dyskretnych.
    
    Argumenty:
      x (np.array): Pierwszy sygnał (wejście).
      h (np.array): Drugi sygnał (odpowiedź impulsowa).
      
    Zwraca:
      np.array: Wynik splotu y[n].
    """
    N_x = len(x)
    N_h = len(h)
    
    # Długość wyniku splotu to N_x + N_h - 1
    N_y = N_x + N_h - 1
    
    # Inicjalizacja tablicy wynikowej zerami
    y = np.zeros(N_y)
    
    # Główna pętla dla zmiennej n (indeks wyjściowy)
    for n in range(N_y):
        
        # Pętla dla sumowania zmiennej k (wzór: y[n] = suma x[k] * h[n - k])
        # Wewnętrzna pętla wykonuje odwrócenie i przesunięcie sygnału h[k]
        for k in range(N_x):
            
            # Indeks dla h to (n - k). Musi mieścić się w granicach h.
            h_index = n - k
            
            # Sprawdzenie, czy indeks h jest prawidłowy (tj. h[n-k] jest niezerowe)
            if (h_index >= 0) and (h_index < N_h):
                y[n] += x[k] * h[h_index]
                
    return y

# --- 2. Przykładowe sygnały ---

# Sygnał wejściowy (prostokątny impuls)
x = np.array([1.0, 2.0, 3.0, 4.0])

# Odpowiedź impulsowa (proste uśrednianie)
h = np.array([0.5, 0.5])

# --- 3. Obliczenia i porównanie ---

# a) Obliczenie za pomocą własnej funkcji
y_manual = my_convolution_sum(x, h)

# b) Obliczenie za pomocą wbudowanej funkcji NumPy
y_numpy = np.convolve(x, h)

# Utworzenie indeksów dla wyniku splotu
N_y = len(y_manual)
n_indices = np.arange(N_y)

# --- 4. Prezentacja wyników ---

print("Sygnał wejściowy x[n]:", x)
print("Odpowiedź impulsowa h[n]:", h)
print("-" * 40)
print("Wynik (własna funkcja):", y_manual)
print("Wynik (numpy.convolve):", y_numpy)
print("-" * 40)
# Sprawdzenie, czy wyniki są identyczne (pomijając ewentualne błędy zaokrągleń)
print("Czy wyniki są identyczne (np.allclose)?", np.allclose(y_manual, y_numpy))


# --- 5. Wizualizacja (Opcjonalnie) ---
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.stem(np.arange(len(x)), x, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title('$x[n]$ (Sygnał wejściowy)')
plt.ylabel('Amplituda')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(np.arange(len(h)), h, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.title('$h[n]$ (Odpowiedź impulsowa)')
plt.ylabel('Amplituda')
plt.grid(True)

plt.subplot(3, 1, 3)
# Wykres własnej implementacji
plt.stem(n_indices, y_manual, linefmt='g-', markerfmt='go', basefmt=" ", label='Własny Splot')
# Nakładanie wyniku numpy jako linii przerywanej (dla weryfikacji)
plt.plot(n_indices, y_numpy, 'kx--', alpha=0.5, label='numpy.convolve') 
plt.title('$y[n] = x[n] * h[n]$ (Wynik Splotu)')
plt.xlabel('n (indeks próbki)')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()