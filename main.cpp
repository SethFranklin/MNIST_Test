
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

typedef unsigned char byte;

const int NUM_IMAGES_TO_TRAIN = 600;

const int IMAGE_OFFSET = 16;
const int LABEL_OFFSET = 8;

const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;

const std::string TRAIN_LABEL_FILE = "train-labels-idx1-ubyte";
const std::string TRAIN_IMAGE_FILE = "train-images-idx3-ubyte";
const std::string TEST_LABEL_FILE = "t10k-labels-idx1-ubyte";
const std::string TEST_IMAGE_FILE = "t10k-images-idx3-ubyte";

unsigned int NumThreads;

struct Image
{

	byte Number;
	byte ImageData[IMAGE_WIDTH * IMAGE_HEIGHT];

};

Image TrainingImages[NUM_IMAGES_TO_TRAIN];

void ConsoleOutputImage(Image* ToDraw)
{

	byte n;

	for (int y = 0; y < IMAGE_HEIGHT; y++)
	{

		for (int x = 0; x < IMAGE_WIDTH; x++)
		{

			n = ToDraw->ImageData[(IMAGE_WIDTH * y) + x];
			if (n > 200) std::cout << "▓";
			else if (n > 150) std::cout << "▒";
			else if (n > 100) std::cout << "░";
			else std::cout << " ";

		}

		std::cout << std::endl;

	}

	std::cout << "Number: " << int(ToDraw->Number) << std::endl;

}

double Sigmoid(double x)
{

	return 1.0 / (1 + exp(-x));

}

struct Network
{

	int NumTrained;
	int NumHiddenLayers; // not including output layer of ten neurons
	int NumNeurons; // neurons per hidden layer

	int NumConstants;

	double GradientStep;
	double GradientMult;

	std::vector<double> Constants;

	Network(int numlayers, int numneurons, double grad1, double grad2)
	{

		NumTrained = 0;

		NumHiddenLayers = numlayers;
		NumNeurons = numneurons;

		GradientStep = grad1;
		GradientMult = grad2;

		NumConstants = (IMAGE_WIDTH * IMAGE_HEIGHT * NumNeurons) + ((NumHiddenLayers - 1) * NumNeurons * NumNeurons) + (NumNeurons * 10) + (NumHiddenLayers * NumNeurons) + 10;

		Constants = std::vector<double>(NumConstants);

	}

};

std::vector<double> EvaluateNetwork(Network* net, Image* img, int GradientConstant)
{

	std::vector<double> evaled(net->NumNeurons);
	std::vector<double> eval(net->NumNeurons);
	std::vector<double> constants(net->Constants);
	double sum;

	if (GradientConstant >= 0) constants[GradientConstant] += net->GradientStep;

	for (int i = 0; i < net->NumNeurons; i++) // first hidden
	{

		sum = constants[i * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)]; // bias

		for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++)
		{

			sum += (double(img->ImageData[j]) / 255.0) * constants[(i * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)) + j]; // weights

		}

		evaled[i] = Sigmoid(sum);

	}

	for (int i = 0; i < net->NumHiddenLayers - 1; i++) // middle hidden
	{

		for (int j = 0; j < net->NumNeurons; j++)
		{

			sum = constants[(net->NumNeurons * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)) + (i * net->NumNeurons * (net->NumNeurons + 1)) + (j * (net->NumNeurons + 1))];

			for (int k = 0; k < net->NumNeurons; k++)
			{

				sum += evaled[k] * constants[(net->NumNeurons * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)) + (i * net->NumNeurons * (net->NumNeurons + 1)) + (j * (net->NumNeurons + 1)) + k];

			}

			eval[j] = Sigmoid(sum); // @TODO: this only works for  1 or two hidden layers, need to alternate between eval and evaled

		}

	}

	evaled = std::vector<double>(10);

	for (int i = 0; i < 10; i++) // output layer
	{

		sum = constants[(net->NumNeurons * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)) + ((net->NumHiddenLayers - 1) * net->NumNeurons * (net->NumNeurons + 1)) + (i * (net->NumNeurons + 1))];

		for (int j = 0; j < net->NumNeurons; j++)
		{

			sum += eval[j] * constants[(net->NumNeurons * ((IMAGE_WIDTH * IMAGE_HEIGHT) + 1)) + ((net->NumHiddenLayers - 1) * net->NumNeurons * (net->NumNeurons + 1)) + (i * (net->NumNeurons + 1)) + j];

		}

		evaled[i] = Sigmoid(sum);

	}

	return evaled;

}

std::mutex gradient_mutex; // find a way to pass these through the thread function?
std::vector<double> gradient;
double initial_cost;
std::vector<double> ideal_out;

void DescentThread(Network* net, int offset)
{

	std::vector<double> partials = std::vector<double>();
	double second_cost;
	double diff;

	for (int i = offset; i < net->NumConstants; i += NumThreads)
	{

		second_cost = 0.0;

		std::vector<double> second_out = EvaluateNetwork(net, &TrainingImages[net->NumTrained], i);

		for (int j = 0; j < 10; j++)
		{

			diff = second_out[j] - ideal_out[j];
			second_cost += (diff * diff);

		}

		partials.push_back((second_cost - initial_cost) / net->GradientStep);

	}

	gradient_mutex.lock();

	for (int i = 0; i < partials.size(); i++) gradient[offset + (NumThreads * i)] = partials[i];

	gradient_mutex.unlock();

}

bool GradientDescent(Network* net)
{

	std::cout << net->NumTrained << "/" << NUM_IMAGES_TO_TRAIN << std::endl;

	std::vector<double> initial_out = EvaluateNetwork(net, &TrainingImages[net->NumTrained], -1);
	ideal_out = std::vector<double>(10);
	ideal_out[TrainingImages[net->NumTrained].Number] = 1.0;
	gradient = std::vector<double>(net->NumConstants);

	initial_cost = 0.0;
	double diff;

	for (int i = 0; i < 10; i++)
	{

		diff = initial_out[i] - ideal_out[i];
		initial_cost += (diff * diff);

	}

	std::vector<std::thread> threads(NumThreads);

	for (int i = 0; i < NumThreads; i++)
	{

		threads[i] = std::thread(DescentThread, net, i);

	}

	for (int i = 0; i < NumThreads; i++)
	{

		threads[i].join();

	}

	for (int i = 0; i < net->NumConstants; i++)
	{

		net->Constants[i] -= net->GradientMult * gradient[i];

	}

	net->NumTrained++;

	if (net->NumTrained >= NUM_IMAGES_TO_TRAIN) return false;
	else return true;

}

int main()
{

	NumThreads = std::thread::hardware_concurrency();

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

		TrainingImages[i].Number = LabelBuffer[i + LABEL_OFFSET];
		std::copy(ImageBuffer + IMAGE_OFFSET + (i * IMAGE_WIDTH * IMAGE_HEIGHT), ImageBuffer + IMAGE_OFFSET + ((i + 1) * IMAGE_WIDTH * IMAGE_HEIGHT), TrainingImages[i].ImageData);

	}

	Network test = Network(2, 16, 0.1, 0.1);

	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;

	t1 = std::chrono::high_resolution_clock::now();

	while (GradientDescent(&test)); // training

	t2 = std::chrono::high_resolution_clock::now();

	std::vector<double> result;

	for (int j = 0; j < 30; j++)
	{

		result = EvaluateNetwork(&test, &TrainingImages[j], -1);

		ConsoleOutputImage(&TrainingImages[j]);

		for (int i = 0; i < result.size(); i++) std::cout << i << ": " << result[i] << std::endl;

	}

	std::cout << "Elapsed time: " << std::chrono::duration<double>(t2 - t1).count() << " sec" << std::endl;

	delete[] ImageBuffer;
	delete[] LabelBuffer;

	return 0;

}