#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

#include "crow.h"
#include "nlohmann/json.hpp"
#include <pqxx/pqxx>

#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using json = nlohmann::json; 
namespace fs = std::filesystem;

string fdmodel_path;


bool loadModel() {
    fdmodel_path = "dnn_model/face_detection_yunet_2023mar.onnx";
    if (fdmodel_path.empty()) {
        cout << "Модель для обнаружения лиц не загружена" << endl;
        return false;
    }
    cout << "Модель для обнаружения лиц загружена" << endl;
    return true;
}

std::vector <char> read_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Ошибка при открытии файла: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Ошибка при чтении файла: " + filename);
    }
    return buffer;

}

void inserting(pqxx::connection& C, const std::string& operation, 
    const std::vector<char>& source, const std::string& source_path,
    const std::vector<char>& image2, const std::string& image2_path,
    const std::vector<char>& result, const std::string& result_path,
    int merge_value, int faces) {

    try {
        pqxx::work transaction(C);

        if (operation == "resize") {
            transaction.exec_params("INSERT INTO resize (source, source_path, result, result_path) VALUES ($1, $2, $3, $4)", pqxx::binarystring(source.data(), source.size()), source_path, pqxx::binarystring(result.data(), result.size()), result_path);
        }
        else if (operation == "overlay") {
            transaction.exec_params("INSERT INTO overlay (source, source_path, image2, image2_path, merge_value, result, result_path) VALUES ($1, $2, $3, $4, $5, $6, $7)", pqxx::binarystring(source.data(), source.size()), source_path, pqxx::binarystring(image2.data(), image2.size()), image2_path, merge_value, pqxx::binarystring(result.data(), result.size()), result_path);
        }
        else if (operation == "finding") {
            transaction.exec_params("INSERT INTO face_finding (source, source_path, faces, result, result_path) VALUES ($1, $2, $3, $4, $5)", pqxx::binarystring(source.data(), source.size()), source_path, faces, pqxx::binarystring(result.data(), result.size()), result_path);
        }
        transaction.commit();
        std::cout << "Результат сохранён в БД." << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Ошибка сохранения в БД:" << e.what() << std::endl;
        throw;
    }

}

string saveImage(const Mat& image, const string& filename, const string& operation_name = "") {
    fs::path newimg_dir = "newimg";
    fs::path output_path = newimg_dir / filename;

    string final_filename;
    if (!operation_name.empty()) {
        fs::path filename_path(filename);
        final_filename = operation_name + "_" + filename_path.stem().string() + filename_path.extension().string();
        output_path = newimg_dir / final_filename;
    }

    try {
        imwrite(output_path.string(), image);
        cout << "Изображение сохранено как: " << output_path.string() << endl;
        return output_path.string();
    }
    catch (const cv::Exception& ex) {
        cerr << "Ошибка сохранения изображения: " << ex.what() << endl;
        throw; 
    }
}

Mat loadImage(const string& file_path) {
    try {
        Mat image = imread(file_path);
        return image;
    }
    catch (const cv::Exception& ex) {
        std::cerr << "Не удалось загрузить изображение: " << ex.what() << std::endl;
        throw;
    }
}

void facefinder(pqxx::connection& C, const string& file_path) {
    string output_path;
    float scoreThreshold = 0.7; //порог уверенности, от 0,1 до 1. выбирать в зависимости от изображений. В данном случае норм от 0.6 до 0.8
    float nmsThreshold = 0.3;
    int topK = 5000;

    Mat image = loadImage(file_path);

    int thick;

    if (image.rows >= 1920){
        thick = image.rows / 640;
    }else thick = 3;

    int core;

    if (image.rows >= 1920){
        core = (image.rows / 19) | 1;
    }else core = 101; //101 достаточно для изображения ХХХХх1920, а значит и для меньших изображений

    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fdmodel_path, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);

    detector->setInputSize(image.size());

    Mat faces;
    detector->detect(image, faces);
    

    for (int i = 0; i < faces.rows; i++) {
        Rect box(faces.at<float>(i, 0), faces.at<float>(i, 1), faces.at<float>(i, 2), faces.at<float>(i, 3));
        Mat faceROI = image(box);
        rectangle(image, box, Scalar(0, 255, 0), thick);
        GaussianBlur(faceROI, faceROI, Size(core, core), 0);
    }

    fs::path input_path(file_path);
    std::string output_filename = input_path.filename().string();
    output_path = saveImage(image, output_filename, "faces");

    std::vector <char> source = read_binary_file(file_path);
    std::vector <char> result = read_binary_file(output_path);

    try {
        inserting(C, "finding", source, file_path, {}, {}, result, output_path, 0, faces.rows);
    }
    catch (const std::exception& e) {
        throw;
    }


}

void image_compress(const string& file_path, pqxx::connection& C) {
    string output_path;
    Mat image = loadImage(file_path);
    if (image.empty()) return;
    Mat image_new = image;
    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(85); // размер от 0 до 100


    fs::path input_path(file_path);
    std::string output_filename = input_path.filename().string();
    output_path = saveImage(image_new, output_filename, "resize");

    std::vector <char> source = read_binary_file(file_path);
    std::vector <char> result = read_binary_file(output_path);

    try {
        inserting(C, "resize", source, file_path, {}, {}, result, output_path, 0, 0);
    }
    catch (const std::exception& e) {
        throw;
    }

}

void image_overlay(const string& file_path, const string& file_path2, double merge, pqxx::connection& C) {
    string output_path;
    double alfa = merge / 100;
    double beta = (100 - merge) / 100;
    Mat image_old = loadImage(file_path);
    if (image_old.empty()) return;
    Mat image_new = loadImage(file_path2);
    if (image_new.empty()) return;
    Mat image_result;
    addWeighted(image_old, alfa, image_new, beta, 0.0, image_result);//ф-ия, для наложения изображений
    //(первое изобр, степень прозрачности, второе изображение, степень его прозрачности, сдвиг, изображение выход)
    fs::path input_path(file_path);

    std::string output_filename = input_path.filename().string();
    output_path = saveImage(image_result, output_filename, "overlay");
    std::vector <char> source = read_binary_file(file_path);
    std::vector <char> image2 = read_binary_file(file_path2);
    std::vector <char> result = read_binary_file(output_path);

    try {
        inserting(C, "overlay", source, file_path, image2, file_path2, result, output_path, merge, 0);
    }
    catch (const std::exception& e) {
        throw;
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    std::cout << "Привет Мир!\n";
    
    if (!loadModel()) {
        return 1;
    }

    pqxx::connection C("host=db port=5432 dbname=handler_db user=user password=pass"); // Объявляем C как объект
    try {
        if (C.is_open()) {
            std::cout << "Успешное соединение с БД." << C.dbname() << std::endl;
        }
        else {
            std::cout << "Соединение с БД не установлено." << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    crow::SimpleApp app;

    CROW_ROUTE(app, "/handler")
        .methods("POST"_method)([&C](const crow::request& req) {
            try {
                json body = json::parse(req.body);

                if (!body.contains("image1") || !body.contains("operation")) {
                    return crow::response(400, "Missing required parameters");
                }
                std::string file_path_str = body["image1"].get<string>();
                std::string file_path_str2 = body["image2"].get<string>();
                std::string operation = body["operation"].get<string>();
                double merge_value = body["merge"].get<double>();
                fs::path file_path(file_path_str);
                fs::path file_path2(file_path_str2);

                std::cout << "file_path: " << file_path.string() << std::endl;
                std::cout << "file_path2: " << file_path2.string() << std::endl;
                std::cout << "operation: " << operation << std::endl;
                std::cout << "merge_value: " << merge_value << std::endl;

                if (operation == "Finding") {
                    facefinder(C, file_path.string());
                }
                else if (operation == "Resize") {
                    image_compress(file_path.string(), C);
                }
                else if (operation == "Merging") {
                    image_overlay(file_path.string(), file_path2.string(), merge_value, C);
                }
                else {
                    return crow::response(400, "Unknown operation");
                }
                json response_data = "Succes";
                crow::response res;
                res.set_header("Content-Type", "application/json; charset=utf-8");
                std::cout << response_data.dump() << std::endl;
                res.write(response_data.dump());
                return res;
            }
            catch (const std::invalid_argument& e) {
                return crow::response(400, e.what());
            }
            catch (const cv::Exception& e) {
                return crow::response(500, "Server error:" + string(e.what()));
            }
            catch (const std::exception& e) {
                return crow::response(400, "ERROR with JSON" + string(e.what()));
            }
        });
    app.port(8080).multithreaded().run();//запуск приложения

    return 0;

}


