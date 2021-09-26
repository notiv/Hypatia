import os
from matplotlib import pyplot as plt
from requests.api import post
from alexandria import detection, ocr, post_processing
import warnings
import cv2

warnings.filterwarnings("ignore")

_API_KEY_FILE = "api_key.txt"
_CONFIDENCE = 0.5

class Scanner:
    def __init__(self):
        self.confidence = _CONFIDENCE
        self.api_key = post_processing.get_api_key(_API_KEY_FILE)
        self.classes, self.colors = detection.load_classes("coco.names")
        self.model = detection.load_model(cfg='yolov3.cfg', model='yolov3.weights')
        self.images_paths, self.images_list = detection.load_images()
        outputs_list = [detection.detect(i, self.model) for i in self.images_list]
        self.boxes_positions = [detection.get_boxes(i, o,
                        self.confidence, self.classes, self.colors)
                        for i, o in zip(self.images_list, outputs_list)]

    def export_plot_rectangles(self, outfile):
        for p, i, b in zip(self.images_paths, self.images_list, self.boxes_positions):
            detection.save_img_rectangles(i, b, outfile)

    def export_plot_rectangles_cv(self, outfile):
        for p, i, b in zip(self.images_paths, self.images_list, self.boxes_positions):
            detection.save_img_rectangles_cv(i, b, outfile)


    def scan(self):
        def get_titles(id, img_box, image, path):
            out = {}
            img, box = img_box
            out["id"] = id
            out["image"] = image.copy()
            out["path"] = path
            out["book_image"] = img
            out["book_box"] = box
            out["title_from_ocr"] = " ".join(ocr.image_to_text(img))
            out["cleaned_titles"] = post_processing.clean_up_text(out["title_from_ocr"])
            out["final_titles"] = post_processing.search_book(out["cleaned_titles"], self.api_key)
            return out

        books_text = {}
        for path, image, boxes in zip(self.images_paths, self.images_list, self.boxes_positions):
            books_text[path] = []
            sub_images = ocr.get_boxes_per_image(
                image=ocr.preprocess4ocr(image),
                boxes=boxes)

            for n, i in enumerate(sub_images):
                try:
                    books_text[path].append(get_titles(n, i, image, path))
                except Exception:
                    warnings.warn(f"{path}: {n} box is discarded")
        self.books_text = books_text

    def search(self, str2search):
        boxes = []
        imgs = []
        for k, v in self.books_text.items():
            image = cv2.imread(k)
            #image = v[0]["image"].copy()
            for i in v:
                if i["final_titles"]:
                    loc = [s.lower().find(str2search.lower()) for s in i["final_titles"]]
                    if any(x >= 0 for x in loc):
                        #imgs.append(i["book_img"])
                        #boxes.append(i["book_box"])
                        xy, xy_up = i["book_box"].args4rectangle_cv
                        image = cv2.rectangle(image, xy, xy_up,
                                    color=(0, 255, 0), thickness=10)
            outfile = os.path.splitext(k)[0] + "_found.jpeg"
            cv2.imwrite(outfile, image)

        return outfile


if __name__ == "__main__":
    s = Scanner()
    s.export_plot_rectangles_cv("./test.png")
    s.scan()
    str2search = "Natural Language Processing"
    s.search(str2search)

