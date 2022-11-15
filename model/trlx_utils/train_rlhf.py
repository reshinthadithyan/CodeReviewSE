import trlx
import argparse
import json
import logging
import torch
from tqdm import tqdm
from model.cross_encoder_utils.model import CrossEncoder
from transformers import AutoTokenizer
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# dummy:   {
#     "sub_text": "stringstream parseitemlink(itemlink);\n    uint32 hexform[ITEMLINKGROUPCOUNT",
#     "pre_blocks": [
#       "If you are using the standard library classes of the same name, I would give the following names the correct namespace qualifier: <code>std::string</code>, <code>std::stringstream</code>, <code>std::hex</code>.",
#     ],
#     "mid_blocks": [
#       "uint32 hexform[ITEMLINKGROUPCOUNT] = {};\n",
#       "<code>ebx</code>, <code>edi</code>, <code>ecx</code>, <code>eax</code> are not good variable names, if you can give them more meaningful names, then do.",
#       "    uint32 ecx = (hexform[i] + 1) * hexsum,\n           eax = ecx * hexform[i];\n",
#       "Personally, I think this is clearer:",
#       "    uint32 ecx = (hexform[i] + 1) * hexsum;\n    uint32 eax = ecx * hexform[i];\n"
#     ],
#     "post_blocks": [
#       "The comment is really bad because it talks about <code>hexform[i]^2 + hexform[i] * hexsum</code> whereas <code>ecx</code> gets the value <code>hexform[i] * hexsum + hexsum</code> and <code>eax</code> gets the value <code>hexform[i]^2 * hexsum + hexform[i] * hexsum</code>. I think the comment needs a pair of parentheses if the code is doing what you meant.",
#       "To be robust, you should check whether the parse worked."
#     ],
#     "question_id": "22",
#     "answer_score": "10",
#     "answer_id": "30"
#   },


MAX_LEN = 512 #Dummy Context length
IND_LEN_BLOCKS = int(MAX_LEN/5) #Equally split the 512 tokens between the 4 blocks and question.
def process_aligned_dataset(example):
    """
    Process the aligned dataset to be used for training.
    """
    final_processed_str : str = "[ANSWERSTART]"
    for key in example.keys():
        if key in ["sub_text","pre_blocks","mid_blocks","post_blocks"]:
            final_processed_str  += key + " :" + "".join(example[key])[:IND_LEN_BLOCKS] #Equally split between lines...
    return final_processed_str


def augment_json(path):
    """
    Augment the json file with the processed string.
    """
    with open(path,"r") as f:
        data = json.load(f)
    output_dict = {}
    for example in tqdm(data):
        output_dict[example["id"]] = example
    
    with open(path,"w") as f:
        json.dump(output_dict,f,indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRLX")
    parser.add_argument("--experiment", type=str, default="crossencoder", help="value | crossencoder") #
    parser.add_argument("--aligned_dataset_path",type=str,default="dataset/aligned_data_with_score_and_key.json")
    parser.add_argument("--cleaned_dataset_path",type=str,default="dataset/CodeReviewSE_CrossEncoder.json")
    #
    parser.add_argument("--trlx_config_path",type=str,default="model/trlx_utils/trlx_train_config.yml") #Path to the trlx config file.
    parser.add_argument("--base_model_path",type=str) #finetuned codegen model to generate.
    parser.add_argument("--ce_model_path",type=str) #The CE_Model to be used in reward fn.

    args = parser.parse_args()

    if args.experiment == "crossencoder":
        class RewardDataset:
            def __init__(self,review_dict_path:str,aligned_dataset_path:str):
                self.dataset = json.load(open(review_dict_path,"r"))
                self.aligned_dataset = json.load(open(aligned_dataset_path,"r"))
                logger.info("Sucessfully loaded the dataset")

            def get_question_body(self,question_id):
                for data in self.dataset:
                    if data["id"] == question_id:
                        return data
            def load_datapoints(self):
                for datapoint in tqdm(self.aligned_dataset):
                    question_id = datapoint["question_id"]
                    post = self.get_question_body(question_id)
                    processed_aligned = process_aligned_dataset(datapoint)
                    yield processed_aligned,post["body"],datapoint["answer_score"] #post["boyd"] is the question text(input to the model)


        augment_json(args.cleaned_dataset_path)
        reward_dataset = RewardDataset(review_dict_path=args.cleaned_dataset_path,aligned_dataset_path=args.aligned_dataset_path)
        dataset = reward_dataset.load_datapoints()

        ce_model = CrossEncoder(args.config_path).load_state_dict(torch.load(os.path.join(args.ce_model_path,"ce_model.pt")))
        ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_model_path)


        def reward_fn(question,answer):
            
            return ce_model(ce_tokenizer("[QUESTIONSTART]"+question+"[ANSWERSTART]"+answer)) #returns a floating value

    trl_config = trlx.TRLConfig.load_yaml("config/trlx_ppo_config.yml")

    model = trlx.train(
        args.base_model_path,
        reward_fn=reward_fn,
        prompts=dataset,
        config=trl_config
    )
    #save the model...  