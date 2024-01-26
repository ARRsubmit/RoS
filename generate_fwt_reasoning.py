
import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def get_dataset_order(dataset_id):
    if dataset_id == 1:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    elif dataset_id == 2:
        dataset_order = ["['sgd_hotels_4']", "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']",
                         "['sgd_media_2']", "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_trains_1']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_flights_1']",
                         "['sgd_services_4']", "['sgd_homes_1']", "['sgd_hotels_1']"]
    elif dataset_id == 3:
        dataset_order = ["['sgd_services_4']", "['sgd_hotels_3']", "['sgd_music_1']", "['sgd_flights_1']",
                         "['sgd_hotels_1']", "['sgd_hotels_4']", "['sgd_media_2']", "['sgd_flights_3']",
                         "['sgd_trains_1']", "['sgd_homes_1']", "['sgd_restaurants_1']", "['sgd_rentalcars_2']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_rentalcars_3']"]
    elif dataset_id == 4:
        dataset_order = ["['sgd_hotels_1']", "['sgd_media_2']", "['sgd_homes_1']", "['sgd_music_1']",
                         "['sgd_services_4']", "['sgd_restaurants_1']", "['sgd_flights_1']", "['sgd_hotels_4']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_trains_1']",
                         "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']"]
    elif dataset_id == 5:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_3']", "['sgd_homes_1']", "['sgd_flights_1']",
                         "['sgd_music_1']", "['sgd_services_3']", "['sgd_rentalcars_3']", "['sgd_media_2']",
                         "['sgd_restaurants_1']", "['sgd_hotels_1']", "['sgd_rentalcars_2']", "['sgd_hotels_4']",
                         "['sgd_hotels_3']", "['sgd_homes_2']", "['sgd_trains_1']"]
    elif dataset_id == 6:
        dataset_order = ["['sgd_restaurants_1']", "['sgd_services_3']", "['sgd_flights_1']", "['sgd_trains_1']",
                         "['sgd_hotels_1']", "['sgd_services_4']", "['sgd_hotels_3']", "['sgd_rentalcars_2']",
                         "['sgd_flights_3']", "['sgd_hotels_4']", "['sgd_homes_2']", "['sgd_homes_1']",
                         "['sgd_rentalcars_3']", "['sgd_media_2']", "['sgd_music_1']"]

    elif dataset_id == 99:
        # debug
        dataset_order = ["['sgd_hotels_4']", "['sgd_trains_1']"]

    else:
        if dataset_id >= 100 and dataset_id <= 114:
            dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                             "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                             "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                             "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
            dataset_order = [dataset_order[dataset_id-100]]
        else:
            raise

    dataset_order2 = []
    for dddd in dataset_order:
        dataset_order2.append(dddd[2:-2])
    print('dataset order')
    print(dataset_order2)
    return dataset_order2



def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    output_file: str = "",
    dataset_id: int = 1, # 1 - 5  
    service_begin_id: int = 1 ,
):
    print(f"dataset_id: {dataset_id}")
    
    dataset_order = get_dataset_order(dataset_id)
    last_service_name = dataset_order[service_begin_id-1]
    
    lora_weights = os.path.join("./checkpoint_files", "Reasoning_LLaMa2-70B_dataset_id_"+str(dataset_id), str(service_begin_id-1)+"-"+last_service_name)
    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)   
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='xxx'"


    output_dir = os.path.join("./output", "Reasoning_LLaMa2-70B_dataset_id_"+str(dataset_id)+"_fwt",)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")
    
    
    
    prompter = Prompter(prompt_template)
    #tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = LlamaTokenizer.from_pretrained('/your_model_path/llama-7b-hf')

    #print(torch.cuda.is_available())
    #sys.exit(1)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            '/your_model_path/llama-7b-hf',
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            '/your_model_path/llama-7b-hf',
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            '/your_model_path/llama-7b-hf', device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    #print(model.config.use_cache)
    #sys.exit(1)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.02,
        top_p=0,
        top_k=1,
        num_beams=1,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        #print(generation_output)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
        #return tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #return output.split("### Response:")[1].strip()



    service_id = service_begin_id
    print(f"current service name: {dataset_order[service_id]}... begin generating!")
    
    output_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")
    print(f"output filename: {output_file}")
    
    testfile_idx = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test.idx"
    testfile_name = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test-LLM.json"
    
    print(f"test filename: {testfile_name}")
    

    if not os.path.isfile(output_file): 
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 
        
    else: 
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        result_out = open(output_file, "a", encoding='utf-8')
    
    idx_lines = open(testfile_idx).readlines()
    data = json.load(open(testfile_name)) 
    for idx_ in range(begin_id, len(data)):
        sample = data[idx_]
        idx_line = idx_lines[idx_].strip()

        Response_list = []

        Response = evaluate(instruction = sample['instruction'], input = sample['input'])
        Response_list.append(Response)

        #print("Input:", input2)
        print("Response list:", Response_list)
        print("Ground truth:", sample['output'])
        print()

        result_out.write(idx_line + "|||" + str(Response_list))
        result_out.write("\n")

        #break
    result_out.close()

if __name__ == "__main__":
    fire.Fire(main)
