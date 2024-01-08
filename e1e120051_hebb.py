import numpy as np

def hebb_learning(input_data1, input_data2):
    # Inisialisasi bobot awal dengan nol
    num_inputs = len(input_data1)
    weights = np.zeros(num_inputs)
    # Set aktivasi unit keluaran y = t (nilai target secara acak -1 atau 1)
    y = np.random.choice([-1, 1])
    # Proses pembelajaran dengan aturan Hebb
    for i in range(num_inputs):
        # Set aktivasi unit masukan Xi = Si
        xi = input_data1[i]

        # Perbaiki bobot menurut persamaan Wi (baru) = Wi(lama) + delta_W
        delta_W = xi * y
        weights[i] += delta_W

    return weights, y

def activation_function(x):
    # Fungsi aktivasi: 1 jika x > 0, -1 jika x <= 0
    return 1 if x > 0 else -1

print("Implementasi Hebb Rule pada Pengenalan Pola")
# Meminta pengguna untuk memasukkan dua pola biner sebagai data training
pattern1 = np.array([int(input(f"Masukkan nilai {i+1} dari data training 1 (-1 atau 1): ")) for i in range(9)])
print("\n")
pattern2 = np.array([int(input(f"Masukkan nilai {i+1} dari data training 2 (-1 atau 1): ")) for i in range(9)])

# Proses pembelajaran dengan aturan Hebb untuk data training 1 dan data training 2
hasil_bobot1, target1 = hebb_learning(pattern1, pattern1)
hasil_bobot2, target2 = hebb_learning(pattern2, pattern2)

# Jumlahkan hasil bobot dari data training 1 dan data training 2
hasil_bobot_terbaru = hasil_bobot1 + hasil_bobot2

# Cetak hasil bobot dan target untuk pola 1 dan pola 2
print("\nTarget untuk pola 1:", target1)
print("Bobot hasil pembelajaran pola 1:", hasil_bobot1)
print("\nTarget untuk pola 2:", target2)
print("Bobot hasil pembelajaran pola 2:", hasil_bobot2)

# Cetak hasil bobot terbaru setelah dijumlahkan
print("\nBobot terbaru setelah dijumlahkan:", hasil_bobot_terbaru)

# pengguna memasukkan pola baru untuk diuji
input_data_new = np.array([int(input(f"Masukkan nilai {i+1} dari pola baru (-1 atau 1): ")) for i in range(9)])

# Hitung hasil akumulasi dengan bobot terbaru dan terapkan fungsi aktivasi
result = np.dot(hasil_bobot_terbaru, input_data_new)
output_activation = activation_function(result)
print("\nHasil akhir dengan fungsi aktivasi (pola baru):", output_activation)

#NAMA : SITTI NUR HALIZA
#NIM  : E1E120051