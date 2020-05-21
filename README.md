# Face-Recognize
Mendeteksi Wajah Seseorang

# Intro
Program ini dibuat untuk mendeteksi wajah-wajah yang sudah di training, oleh karena itu dibutuhkan beberapa gambar untuk mendeteksi wajah.

# Tahap Penggunaan Program
1. Install Python melalui http://www.python.org/downloads/ dan lakukan Proses Installasi.

### Saya disini menggunakan Pycharm, jika ingin menggunakan IDE pycharm dapat mengikuti langkah dibawah ini
2. Install melalui https://www.jetbrains.com/pycharm/download/
3. Pilih Versi Community (Gratis) dan lakukan Proses Installasi.

### langkah selanjutnya, menuliskan code.
4. pada program pycharm, buat lah project baru dengan nama project sesuka kalian (saya menggunakan nama project FaceRecognition)
5. Setelah project terbuka dengan sempurna, buatlah dua folder, dengan nama masing-masing "dataset" dan "trainer"
* folder ini berfungsi untuk menyimpan data gambar dan data training jadi kosongkan dan jangan diisi.
6. tambahkan file "haarcascade_frontalface_default.xml" pada project, sesuaikan dengan lokasi project yang kalian buat, saya membuatnya 
    di C:\Users\{{ nama user }}\Desktop\FaceRecognition
7. install beberapa depedency melalui terminal dengan mengetikan :
* pip install opencv-contrib-python
* pip install pillow
* install 1 per 1 hingga selesai
8. Jalankan file dataset.py terlebih dahulu, code pada dataset akan mengambil 30 gambar
9. lalu jalankan code training pada training.py, code training akan mempelajari wajah yang tersimpan pada folder dataset dan menyimpannya di traning/training.yml
10. jalankan recognize.py, hasil dari dataset dan training akan muncul pada pada program ini.
