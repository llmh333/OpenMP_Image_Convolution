#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>    
#include <omp.h>     

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace std;
namespace fs = std::filesystem;
using namespace std::chrono;

struct Image {
    int width, height, channels;
    vector<unsigned char> pixels;
};

Image loadImage(const string& filename) {
    Image img;
    unsigned char* data = stbi_load(filename.c_str(), &img.width, &img.height, &img.channels, 1);
    if (!data) {
        img.width = 0; 
        return img;
    }
    img.pixels.assign(data, data + img.width * img.height);
    stbi_image_free(data);
    return img;
}

void saveImage(const string& filename, const Image& img) {
    stbi_write_jpg(filename.c_str(), img.width, img.height, 1, img.pixels.data(), 100);
}

const int KERNEL_BLUR[3][3] = {
    { 1, 2, 1 },
    { 2, 4, 2 },
    { 1, 2, 1 }
};

const int KERNEL_SHARPEN[3][3] = {
    {-1, -1, -1 },
    {-1,  9, -1 },
    {-1, -1, -1 }
};

// --- HÀM 1: XỬ LÝ TUẦN TỰ (KHÔNG DÙNG OPENMP) ---
void processSequential(const Image& input, Image& output, const int kernel[3][3], int divisor) {
    int w = input.width; int h = input.height;
    output.width = w; output.height = h; output.channels = 1;
    output.pixels.resize(w * h);

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += input.pixels[(y + ky) * w + (x + kx)] * kernel[ky + 1][kx + 1];
                }
            }
            sum /= divisor;
            if (sum < 0) sum = 0; if (sum > 255) sum = 255;
            output.pixels[y * w + x] = (unsigned char)sum;
        }
    }
}

// --- HÀM 2: XỬ LÝ SONG SONG (DÙNG OPENMP) ---
void processOpenMP(const Image& input, Image& output, const int kernel[3][3], int divisor) {
    int w = input.width; int h = input.height;
    output.width = w; output.height = h; output.channels = 1;
    output.pixels.resize(w * h);

    #pragma omp parallel for collapse(2) \
        schedule(static) \
        shared(input, output, kernel, divisor, w, h)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += input.pixels[(y + ky) * w + (x + kx)] * kernel[ky + 1][kx + 1];
                }
            }
            sum /= divisor;
            if (sum < 0) sum = 0; if (sum > 255) sum = 255;
            output.pixels[y * w + x] = (unsigned char)sum;
        }
    }
}

int main() {
    int num_thread;
    cout << "So luong muon chay: ";
    cin >> num_thread;
    omp_set_num_threads(num_thread);
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "--- CAU HINH OPENMP ---" << endl;
            cout << "So luong luong (Threads) dang chay: " << omp_get_max_threads() << endl;
            cout << "So luong vi xu ly (Procs) co san: "   << omp_get_num_procs() << endl;
            cout << "-----------------------" << endl;
        }
    }

    string inputFolder = "data";
    string outSeqFolder = "output_sequential";
    string outOmpFolder = "output_openmp";

    if (!fs::exists(inputFolder)) {
        cerr << "Loi: Khong tim thay thu muc 'data'" << endl;
        return 1;
    }
    if (!fs::exists(outSeqFolder)) fs::create_directory(outSeqFolder);
    if (!fs::exists(outOmpFolder)) fs::create_directory(outOmpFolder);

    double totalTimeSeq = 0;
    double totalTimeOmp = 0;
    int fileCount = 0;

    cout << "==================== BAT DAU BENCHMARK ====================" << endl;
    cout << setw(40) << left << "File" << setw(15) << left << "Tuan tu(s)" << setw(15) << left << "OpenMP(s)" << setw(10) << left << "Tang toc" << endl;
    cout << "-----------------------------------------------------------" << endl;

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        string filename = entry.path().filename().string();
        string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

        Image imgIn = loadImage(entry.path().string());
        if (imgIn.width == 0) continue;

        Image imgBlur, imgFinal;
        
        auto start = high_resolution_clock::now();
        processSequential(imgIn, imgBlur, KERNEL_BLUR, 16);
        processSequential(imgBlur, imgFinal, KERNEL_SHARPEN, 1);
        auto end = high_resolution_clock::now();

        duration<double> diffSeq = end - start;
        totalTimeSeq += diffSeq.count();
        
        saveImage(outSeqFolder + "/" + "result_" + filename, imgFinal);


        start = high_resolution_clock::now();
        processOpenMP(imgIn, imgBlur, KERNEL_BLUR, 16);
        processOpenMP(imgBlur, imgFinal, KERNEL_SHARPEN, 1);
        end = high_resolution_clock::now();

        duration<double> diffOmp = end - start;
        totalTimeOmp += diffOmp.count();

        saveImage(outOmpFolder + "/" + "result_" + filename, imgFinal);

        cout << setw(40) << left << filename 
             << setw(15) << left << diffSeq.count()
             << setw(15) << left << diffOmp.count()
             << "x" << (diffSeq.count() / diffOmp.count()) << endl;

        fileCount++;
    }

    cout << "-----------------------------------------------------------" << endl;
    cout << "TONG KET (" << fileCount << " file):" << endl;
    cout << "Tong thoi gian Tuan tu: " << totalTimeSeq << " giay" << endl;
    cout << "Tong thoi gian OpenMP : " << totalTimeOmp << " giay" << endl;
    if (totalTimeOmp > 0)
        cout << "=> TOC DO TRUNG BINH TANG GAP: " << totalTimeSeq / totalTimeOmp << " LAN!" << endl;

    return 0;
}