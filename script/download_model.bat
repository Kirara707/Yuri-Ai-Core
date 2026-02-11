@echo off
echo === Downloading pre-trained weights ===
echo Please download model manually from HuggingFace:
echo https://huggingface.co/yeyeye0118/BERT-Yuri-CLS-Large
exit /b
powershell -command "Expand-Archive -Force models\BERT-Yuri-CLS.zip models\"
del models\BERT-Yuri-CLS.zip
echo Done!
