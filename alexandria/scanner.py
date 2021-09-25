from matplotlib import pyplot as plt
from requests.api import post
from alexandria import detection, ocr, post_processing
import warnings

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
        def get_titles(id, img):
            out = {}
            out["id"] = id
            out["book_image"] = img
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
                    books_text[path].append(get_titles(n, i))
                except Exception:
                    warnings.warn(f"{path}: {n} box is discarded")

        return books_text


if __name__ == "__main__":
    s = Scanner()
    s.export_plot_rectangles_cv("./test.png")