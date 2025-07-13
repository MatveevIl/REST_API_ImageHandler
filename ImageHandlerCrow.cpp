#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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

/*CascadeClassifier face_cascade; //объявляем объект стандартного для опенСВ класса CascadeClassifier для обнаружения объектов
CascadeClassifier face_cascade2;
CascadeClassifier face_cascade3;*/

//bool cascade_loaded = false;

/*bool loadCascade(CascadeClassifier& cascade, const string& filename) {
    if (!cascade.load(filename)) {
        cout << "Не удалось загрузить классификатор каскадов Хаара" << endl;
        return false;
    }
    cout << "Удалось загрузить классификатор каскадов Хаара" << endl;
    return true;
}*/
/*bool loadCascades() {
    bool all_loaded = true;
    all_loaded &= loadCascade(face_cascade, "haarcascades/haarcascade_frontalface_default.xml");
    all_loaded &= loadCascade(face_cascade2, "haarcascades/haarcascade_frontalface_alt.xml");
    all_loaded &= loadCascade(face_cascade3, "haarcascades/haarcascade_frontalface_alt2.xml");
    cascade_loaded = all_loaded;
    return all_loaded;
}*/

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
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Could not read file: " + filename);
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

    // Добавляем префикс с названием операции, если он указан
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
        throw; //Выкидываем исключение для Crow
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
    float scale = 1.0;
    float scoreThreshold = 0.9;
    float nmsThreshold = 0.3;
    int topK = 5000;

    Mat image = loadImage(file_path);

    int imageWidth = int(image.cols * scale);
    int imageHeight = int(image.rows * scale);
    resize(image, image, Size(imageWidth, imageHeight));

    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(fdmodel_path, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);

    detector->setInputSize(image.size());

    // Detect faces
    Mat faces;
    detector->detect(image, faces);
    
    int thick = 3;

    for (int i = 0; i < faces.rows; i++) {
        rectangle(image, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(255, 0, 0), thick);
    }

    fs::path input_path(file_path);
    std::string output_filename = input_path.filename().string();
    output_path = saveImage(image, output_filename, "faces");

    std::vector <char> source = read_binary_file(file_path);
    std::vector <char> result = read_binary_file(output_path);

    try {
        inserting(C, "finding", source, file_path, {}, {}, result, output_path, 0, 0);
    }
    catch (const std::exception& e) {
        throw;
    }


}


/*void face_finder(const string& file_path, pqxx::connection& C) {
    string output_path;
    Mat image = loadImage(file_path);
    if (image.empty()) return;
    Mat greyImage; //Поиск объектов с помощью каскадов Хаара работает лучше с серым изображением
    cvtColor(image, greyImage, COLOR_BGR2GRAY);
    std::vector<Rect> faces; //объявляем faces = вектор(массив, изменяющийся динамически)  для хранения прямоугольников = Rect
    std::vector<Rect> faces2;
    std::vector<Rect> faces3;
    {
        if (!cascade_loaded) {
            std::cerr << "Ошибка. Не загружены каскады" << std::endl;
        }
        face_cascade.detectMultiScale(greyImage, faces, 1.05, 6, 0, Size(35, 40));
        /*функция для обнаружения лиц(входное изображение; вектор, куда сохраняем найденные лица;
        первый параметр в диапазоне 1,05-1,4 - параметр масштабирования, чем меньше - тем точнее, но дольше работает;
        второй параметр 2-6 - параметр, определяющий, сколько прямоугольников дб рядом, чтоб кандидат стал лицом, чем больше значение - тем меньше ложных срабатываний, но можно пропустить лица
        0 - других флагов нет
        CASCADE_SCALE_IMAGE - размер изображения масштабируем во время поиска
        Сайз - минимальный размер лица для определения
        face_cascade2.detectMultiScale(greyImage, faces2, 1.05, 5, 0, Size(30, 30));
        face_cascade3.detectMultiScale(greyImage, faces3, 1.04, 4, 0, Size(25, 25));
    }
    std::vector<Rect> final_face;
    for (size_t i = 0; i < faces.size(); i++) {
        for (size_t j = 0; j < faces2.size(); j++) {
            for (size_t y = 0; y < faces3.size(); y++) {

                Rect intersection = faces[i] & faces2[j] & faces3[y];
                if (intersection.area() > 0) {
                    final_face.push_back(faces[i]);
                    break;
                }
            }
        }
    }
    for (size_t i = 0; i < final_face.size(); i++) {
        rectangle(image, final_face[i], Scalar(255, 0, 0), 2);
        Mat faceROI = image(final_face[i]); //создаёт объект, типа Mat, являющийся регионом интереса
        GaussianBlur(faceROI, faceROI, Size(101, 101), 0); //фэйсРои - лицо входящее, фэйсРОИ - лицо выходящее; размер ядра Гаусса и отклонение по Х и У - 0(стандартное)
        faceROI.copyTo(image(final_face[i])); //копирует размытое на исходное изображение
    }
    fs::path input_path(file_path);
    std::string output_filename = input_path.filename().string();
    output_path = saveImage(image, output_filename, "faces");

    std::vector <char> source = read_binary_file(file_path);
    std::vector <char> result = read_binary_file(output_path);

    try {
        inserting(C, "finding", source, file_path, {}, {}, result, output_path, 0, final_face.size());
    }
    catch (const std::exception& e) {
        throw;
    }

}*/

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
    
    
    /*if (!loadCascades()) {
        return 1; 
    }*/
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
                    //face_finder(file_path.string(), C);
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


