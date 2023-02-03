import os
import json
from collections import defaultdict  # this is std lib
from argparse import ArgumentParser
from PIL import Image
import clip
import torch
import language_evaluation


def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])

        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results["CIDEr"]


class CLIPScore:
    def __init__(self, deivce):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.0
        # max_score = 0.0
        # min_score = 1.0
        # max_imgname = ""
        # min_imgname = ""
        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")
            clip_score = self.getCLIPScore(image, pred_caption)
            #     if clip_score > max_score:
            #         max_score = clip_score
            #         max_imgname = img_name
            #     elif clip_score < min_score:
            #         min_score = clip_score
            #         min_imgname = img_name

            total_score += clip_score
        # print(f"max score: {max_score} , name:{max_imgname}")
        # print(f"min score: {min_score} , name:{min_imgname}")
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        cossim = torch.nn.CosineSimilarity()

        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(caption).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
        score = 2.5 * max(cossim(image_features, text_features), 0)
        return score.squeeze()


def getScore(
    pred_file,
    annotation_file="/home/stan/hw3-stanthemaker/hw3_data/p2_data/val.json",
    images_root="/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/val",
    deivce="cuda",
):
    # Read data
    predictions = readJSON(pred_file)
    annotations = readJSON(annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore
    clip_score = CLIPScore(deivce=deivce)(predictions, images_root)
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")
    return cider_score, clip_score


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file", "-p", help="Prediction json file")
    parser.add_argument(
        "--images_root",
        default="/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/val",
        help="Image root",
    )
    parser.add_argument(
        "--annotation_file",
        default="/home/stan/hw3-stanthemaker/hw3_data/p2_data/val.json",
        help="Annotation json file",
    )

    args = parser.parse_args()

    getScore(args.pred_file, args.annotation_file, args.images_root)
