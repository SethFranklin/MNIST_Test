
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>

typedef unsigned char byte;

const int NUM_IMAGES_TO_TRAIN = 100;

const int IMAGE_OFFSET = 16;
const int LABEL_OFFSET = 8;

const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;

const std::string TRAIN_LABEL_FILE = "train-labels-idx1-ubyte";
const std::string TRAIN_IMAGE_FILE = "train-images-idx3-ubyte";
const std::string TEST_LABEL_FILE = "t10k-labels-idx1-ubyte";
const std::string TEST_IMAGE_FILE = "t10k-images-idx3-ubyte";

struct Image
{

	byte Number;
	byte ImageData[IMAGE_WIDTH * IMAGE_HEIGHT];

	Image(byte n, char * ndata)
	{

		Number = n;
		std::copy(ndata, ndata + (IMAGE_WIDTH * IMAGE_HEIGHT), this->ImageData);

	}

	Image() {}

};

Image TrainingImages[NUM_IMAGES_TO_TRAIN];

void ConsoleOutputImage(Image ToDraw)
{

	byte n;

	for (int y = 0; y < IMAGE_HEIGHT; y++)
	{

		for (int x = 0; x < IMAGE_WIDTH; x++)
		{

			n = ToDraw.ImageData[(IMAGE_WIDTH * y) + x];
			if (n > 200) std::cout << "▓";
			else if (n > 150) std::cout << "▒";
			else if (n > 100) std::cout << "░";
			else std::cout << " ";

		}

		std::cout << std::endl;

	}

	std::cout << "Number: " << int(ToDraw.Number) << std::endl;

}

double Sigmoid(double x)
{

	return 1.0 / (1 + exp(-x));

}

struct Network
{

	int NumTrained;
	int NumHiddenLayers;
	int NumNeurons; // neurons per hidden layer

	int NumConstants;

	std::vector<double> Constants;

	Network(int numlayers, int numneurons)
	{

		NumHiddenLayers = numlayers;
		NumNeurons = numneurons;

		NumConstants = 4;

		Constants = std::vector<double>(NumConstants);

	}

	Network() {}

};

int main()
{

	std::ifstream ImageStream(TRAIN_IMAGE_FILE.c_str(), std::ifstream::binary);
	char * ImageBuffer = new char[(NUM_IMAGES_TO_TRAIN * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET];

	if (ImageStream)
	{

		ImageStream.read(ImageBuffer, (NUM_IMAGES_TO_TRAIN * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET);
		ImageStream.close();

	}
	else std::cout << "Couldn't read " + TRAIN_IMAGE_FILE << std::endl;

	std::ifstream LabelStream(TRAIN_LABEL_FILE.c_str(), std::ifstream::binary);
	char * LabelBuffer = new char[NUM_IMAGES_TO_TRAIN + LABEL_OFFSET];

	if (LabelStream)
	{

		LabelStream.read(LabelBuffer, NUM_IMAGES_TO_TRAIN + LABEL_OFFSET);
		LabelStream.close();

	}
	else std::cout << "Couldn't read " + TRAIN_LABEL_FILE << std::endl;

	for (int i = 0; i < NUM_IMAGES_TO_TRAIN; i++)
	{

		TrainingImages[i] = Image(LabelBuffer[i + LABEL_OFFSET], ImageBuffer + IMAGE_OFFSET + (i * IMAGE_WIDTH * IMAGE_HEIGHT));

	}

	for (int i = 70; i < 100; i++) ConsoleOutputImage(TrainingImages[i]);

	delete[] ImageBuffer;
	delete[] LabelBuffer;

	return 0;

}