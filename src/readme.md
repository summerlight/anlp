This project generates a feature input file compatible to CRF++ or [Wapiti](https://wapiti.limsi.fr/). In the case of Wapiti, you can train a CRF model as below:

    wapiti train -p pattern.txt -2 1 input.txt model.txt

By using the generated model file, now we can label some input.

    wapiti label -m model.txt input.txt result.txt
