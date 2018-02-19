print("Name: Gautam and Sujit")
print("PersonNO: 50245840 and 50247206")
print("1. Extraction")
print("2. Convolution Neural Network")
ch = int(input("Please Enter your choice"))
if(ch==1):
    print("Implementing Extraction ..")
    with open('extraction.py') as source_file:
        exec (source_file.read())
if(ch==2):
    print("Implementing Convolutional Neural Network")
    with open('ConvolutionalNN.py') as source_file:
        exec (source_file.read())
