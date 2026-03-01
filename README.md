This is the whole project folder to create and train your own Language model from scratch (almost)

1. in models/model.py:
   the neural network

2. Data:
   - preProcessed/
     has all the data collector classes and the base tokenizer class. when adding new dataset for pre processing make the class in similar format. will be called in training.
   - raw/
     is for keeping raw .pdf's .srt's .json's and other files to read from if needed.
   - processed/
     for formating data into instruct ready structure.
     structure: {###: [system_prompt]\n##: [User prompt]\n####: [model answer]} // gpt2 tokenizer recognises '##', '###' and '####' as single tokens. so this saves tokens on context compared to using <|system|>, <|user|> and <|assistant|>

3. Training:
   - train.py:
     main training file. change congig according to system. has checkpoint system and gradiant accumilation.
