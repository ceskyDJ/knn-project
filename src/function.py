import pickle
import xxhash

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

with open("../datasets/inputs_for_function/1/encoding.pkl", "rb") as f:
  e = pickle.load(f)
  
with open("../datasets/inputs_for_function/1/preds.pkl", "rb") as f:
  p = pickle.load(f)
  
words = [['@jaxxb966', '2', 'years', 'ago', '"You', 'only', 'find', 'dead', 'people?”', '"They', 'tend', 'to', 'move', 'around', 'a', 'lot', 'less”', 'Fucking', 'killed', 'me.', '1', '42k', 'GI', 'Reply', '+', '3replies', '@dolliesmile7799', '2', 'years', 'ago', '(edited)', 'Dorian:', '*opens', 'his', 'mouth*', 'Chetney:', '..s0', 'you', 'have', 'chosen..death', '£429', 'DB', 'reply', '=', '2replies', '@WuxianTee', '1', 'year', 'ago', 'Death', 'by', 'words.', 'Slowly', 'crushing', 'you,', 'like', 'a', 'machine-made', 'toy.', '4', 'GP', 'Reply', '@chriswilder9719', '1', 'year', 'ago', '(edited)', 'Haha', "that's", 'spot', 'on', 'what', 'happened', 'to.', 'Travis', 'said', 'in', 'four', 'sided', 'dive', 'it', 'was', 'the', 'moment', 'Robby', 'opened', 'his', 'mouth', 'he', 'decided', 'Chet', 'would', 'hate', 'Dorian', '1', '3', 'Gl', 'Reply', '@afineegg1040', '2', 'years', 'ago', '(edited)', '&', '|', 'LOVE', 'how', 'much', 'Liam', 'struggles', 'with', 'Chetney', 'because', 'outside', 'of', 'the', 'game', 'everyone', 'thinks', 'Chetney', 'is', 'fucking', 'hilarious,', 'Liam', 'included,', 'but', 'in', 'canon', 'Orym', 'dislikes', 'him', 'at', 'first', 'due', 'to', 'his.', 'abrasive', 'personality', 'and', 'dislike', 'of', 'Dorian,', 'so', 'you', 'see', 'this', 'hilarious', 'combination', 'of', 'an', 'angry', 'glare', 'and', 'a', 'barely-holding-back-laughter', 'expression', 'ty', 'DB', 'reply', '¥', '4replies', '@amalaspina', '2', 'years', 'ago', 'Dorian:', '"You', 'seem', 'to', 'be', 'really', 'into', 'woodwork?"', 'Chetney:', '*...', 'S02!"', '{Im', 'loving', 'the', 'one', 'sided', 'hate', 'between', 'these', 'two', 'already', 'ty', 'DB', 'reply']]
preds = [p]

# print keys
# print(e.keys())
# print(p.keys())

# print first lines of data
# print("e - input_ids: ", e["  "][:3])
# print("e - token_type_ids: ", e["token_type_ids"][:3])
# print("e - attention_mask: ", e["attention_mask"][:3])
# print("e - bbox: ", e["bbox"][:3])
# print("e - image: ", e["image"][:3])

# print("p - true_predictions: ", len(preds[0]["true_predictions"]), preds[0]["true_predictions"])
# print("p - true_start_end_predictions: ", len(preds[0]["true_start_end_predictions"]), preds[0]["true_start_end_predictions"])
# print("p - true_boxes: ", len(preds[0]["true_boxes"]), preds[0]["true_boxes"])

def filter_padding(bounding_boxes, start_end_predictions, predictions):
    filtered_boxes = [] 
    filtered_start_end = []
    filtered_predictions = []
    
    
    iterator = 0
    for box in bounding_boxes:
        if any(coord != 0.0 for coord in box):
            filtered_boxes.append(box)
            filtered_start_end.append(start_end_predictions[iterator])
            filtered_predictions.append(predictions[iterator])
            iterator += 1
        else:
            iterator += 1
            continue

    return filtered_boxes, filtered_start_end, filtered_predictions

for data in preds:
    data["true_boxes"], data["true_start_end_predictions"], data["true_predictions"] = filter_padding(data["true_boxes"], data["true_start_end_predictions"], data["true_predictions"])

# print("words:", len(words[0]), words[0])
# print("p - true_predictions: ", len(preds[0]["true_predictions"]), preds[0]["true_predictions"])
# print("p - true_start_end_predictions: ", len(preds[0]["true_start_end_predictions"]), preds[0]["true_start_end_predictions"])
# print("p - true_boxes: ", len(preds[0]["true_boxes"]), preds[0]["true_boxes"])
# print("\n\n\n")

def calculate_id(identifier, date):
    data = f"{identifier}{date}".encode('utf-8')
    unique_id = xxhash.xxh64(data).hexdigest()
    return unique_id

def process_data(preds, words):
    results = []
        
    idx = 0
    for dictionary in preds:
        true_start_end_prediction = dictionary.get("true_start_end_predictions", [])
        true_predictions = dictionary.get("true_predictions", [])
        true_boxes = dictionary.get("true_boxes", [])
        
        if not true_start_end_prediction:
            print("NOT PREDICTION ERROR")
            exit()
        
        current_result = [] 
        wasStart = False
        
        val_idx = 0
        for value in true_start_end_prediction:
            if val_idx == (len(true_start_end_prediction)-1):
                break
            # print(value, "---", val_idx, "---", true_predictions[val_idx], "---", words[idx][val_idx])
            if value == 0 and wasStart and true_predictions[val_idx] == "Body" or value == 0 and wasStart and true_predictions[val_idx] == "Author":
                current_dict["body"]["boxes"].append(true_boxes[val_idx])
                current_dict["body"]["value"].append(words[idx][val_idx])
            elif value == 0 and wasStart and true_predictions[val_idx] == "Date":
                current_dict["date"]["boxes"].append(true_boxes[val_idx])
                current_dict["date"]["value"].append(words[idx][val_idx])
                
            elif value == 0 and wasStart:
                val_idx += 1 
                continue
            
            if value == 1: 
                # print("TU SOM 2 --- ", val_idx)
                current_dict = {}
                wasStart = True
                current_dict["parent"] = None
                
                current_dict["author"] = {}
                current_dict["author"]["value"] = words[idx][val_idx]
                current_dict["author"]["boxes"] = []
                current_dict["author"]["boxes"].append(true_boxes[val_idx])
                
                current_dict["date"] = {}
                current_dict["date"]["value"] = []
                current_dict["date"]["boxes"] = []
                
                current_dict["body"] = {}
                current_dict["body"]["value"] = []
                current_dict["body"]["boxes"] = []
                            
            if value == 2:
                # print("TU SOM --- ", val_idx)
                wasStart = False
                current_dict["id"] = calculate_id(current_dict["author"],  " ".join(current_dict["date"]))
                
                current_dict["body"]["boxes"].append(true_boxes[val_idx])
                current_dict["body"]["value"].append(words[idx][val_idx])
                
                current_result.append(current_dict)
                    
            val_idx += 1 
            
        results.append(current_result)
        
        idx +=1
    
    return results

result = process_data(preds, words)
# print(result)


####################### SHOW ON IMAGE

img_path = Path("../datasets/inputs_for_function/1/yt_new_big_1.png")

image = Image.open(
        img_path
).convert("RGB")

width, height = image.size

draw = ImageDraw.Draw(image)

font = ImageFont.load_default()

label2color = {'author': 'blue', 'body': 'green', 'date': 'orange', 'O': 'violet'}
se_label = {0: "", 1: " start", 2: " end"}

for input_image in result: 
    for comment in input_image:
        print(comment)
        for key, value in comment.items():
            if key == "parent" or key == "id":
                continue
            label = key
            boxes = value['boxes']
            
            for box in boxes:
                draw.rectangle(box, outline=label2color[label])
                draw.text((box[0] + 10, box[1] - 10), text=label, fill=label2color[label], font=font)

image.show()

