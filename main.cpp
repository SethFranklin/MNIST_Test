
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <math.h>

typedef unsigned char byte;

const int NUM_TRAINING_IMAGES = 60000;
const int NUM_TESTING_IMAGES = 10000;

const int IMAGE_OFFSET = 16;
const int LABEL_OFFSET = 8;

const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;

const std::string TRAIN_LABEL_FILE = "train-labels-idx1-ubyte";
const std::string TRAIN_IMAGE_FILE = "train-images-idx3-ubyte";
const std::string TEST_LABEL_FILE = "t10k-labels-idx1-ubyte";
const std::string TEST_IMAGE_FILE = "t10k-images-idx3-ubyte";

std::chrono::high_resolution_clock::time_point beginning;

unsigned int NumThreads;

struct Image
{

	byte Number;
	byte ImageData[IMAGE_WIDTH * IMAGE_HEIGHT];

};

Image TrainingImages[NUM_TRAINING_IMAGES];
Image TestingImages[NUM_TESTING_IMAGES];

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
	double LearningRate;

	std::vector<double> Constants;

	Network(int numlayers, int numneurons, double grad1, double grad2)
	{

		NumTrained = 0;

		NumHiddenLayers = numlayers;
		NumNeurons = numneurons;

		GradientStep = grad1;
		LearningRate = grad2;

		NumConstants = (IMAGE_WIDTH * IMAGE_HEIGHT * NumNeurons) + ((NumHiddenLayers - 1) * NumNeurons * NumNeurons) + (NumNeurons * 10) + (NumHiddenLayers * NumNeurons) + 10;

		Constants = std::vector<double>(NumConstants);

		// he-et-al normal weights initialization:

		std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now() - beginning;
  		unsigned seed = d.count();

		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0, 1.0);

		double heetalfactor = sqrt(2.0 / double(IMAGE_WIDTH * IMAGE_HEIGHT));

		int c = 0;

		for (int i = 0; i < NumNeurons; i++) // first hidden
		{

			c++;

			for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++)
			{

				Constants[c] = distribution(generator) * heetalfactor;
				c++;

			}

		}

		heetalfactor = sqrt(2.0 / double(NumNeurons));

		for (int i = 0; i < NumHiddenLayers - 1; i++) // middle hidden
		{

			for (int j = 0; j < NumNeurons; j++)
			{

				c++;

				for (int k = 0; k < NumNeurons; k++)
				{

					Constants[c] = distribution(generator) * heetalfactor;
					c++;

				}

			}

		}

		for (int i = 0; i < 10; i++) // output layer
		{

			c++;

			for (int j = 0; j < NumNeurons; j++)
			{

				Constants[c] = distribution(generator) * heetalfactor;
				c++;

			}

		}

	}

};

std::vector<double> EvaluateNetwork(Network* net, Image* img, int GradientConstant)
{

	std::vector<double> evaled(net->NumNeurons);
	std::vector<double> eval(net->NumNeurons);
	std::vector<double> constants(net->Constants);
	double sum;
	double softsum = 0; // for softmax

	int c = 0;

	if (GradientConstant >= 0) constants[GradientConstant] += net->GradientStep;

	for (int i = 0; i < net->NumNeurons; i++) // first hidden
	{

		sum = constants[c];
		c++;

		for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++)
		{

			sum += (double(img->ImageData[j]) / 255.0) * constants[c];
			c++;

		}

		evaled[i] = Sigmoid(sum);

	}

	for (int i = 0; i < net->NumHiddenLayers - 1; i++) // middle hidden
	{

		for (int j = 0; j < net->NumNeurons; j++)
		{

			sum = constants[c];
			c++;

			for (int k = 0; k < net->NumNeurons; k++)
			{

				sum += evaled[k] * constants[c];
				c++;

			}

			eval[j] = Sigmoid(sum); // @TODO: this only works for  1 or two hidden layers, need to alternate between eval and evaled

		}

	}

	evaled = std::vector<double>(10);

	for (int i = 0; i < 10; i++) // output layer
	{

		sum = constants[c];
		c++;

		for (int j = 0; j < net->NumNeurons; j++)
		{

			sum += eval[j] * constants[c];
			c++;

		}

		evaled[i] = Sigmoid(sum);
		softsum += evaled[i];

	}

	//for (int i = 0; i < 10; i++) evaled[i] /= softsum; // softmax

	return evaled;

}

std::mutex gradient_mutex; // find a way to pass these through the thread function?
std::vector<double> gradient;
double initial_cost;
std::vector<double> ideal_out;
std::vector<int> permutation = std::vector<int>(NUM_TRAINING_IMAGES);

void DescentThread(Network* net, int offset)
{

	std::vector<double> partials = std::vector<double>();
	double second_cost;
	double diff;

	for (int i = offset; i < net->NumConstants; i += NumThreads)
	{

		second_cost = 0.0;

		std::vector<double> second_out = EvaluateNetwork(net, &TrainingImages[permutation[net->NumTrained]], i);

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

void GradientDescent(Network* net, int num_to_train)
{

	std::vector<double> initial_out;
	double diff;

	for (int i = 0; i < NUM_TRAINING_IMAGES; i++) permutation[i] = i;

	std::srand(unsigned(std::time(0)));
	std::random_shuffle(permutation.begin(), permutation.end());

	while (net->NumTrained < num_to_train)
	{

		std::cout << net->NumTrained << "/" << num_to_train << std::endl;

		initial_out = EvaluateNetwork(net, &TrainingImages[permutation[net->NumTrained]], -1);
		ideal_out = std::vector<double>(10);
		ideal_out[TrainingImages[permutation[net->NumTrained]].Number] = 1.0;
		gradient = std::vector<double>(net->NumConstants);

		initial_cost = 0.0;

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

			net->Constants[i] -= net->LearningRate * gradient[i];

		}

		net->NumTrained++;

	}

}

int main()
{

	beginning = std::chrono::high_resolution_clock::now();

	NumThreads = std::thread::hardware_concurrency();

	// training images load

	std::ifstream ImageStream(TRAIN_IMAGE_FILE.c_str(), std::ifstream::binary);
	char * ImageBuffer = new char[(NUM_TRAINING_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET];

	if (ImageStream)
	{

		ImageStream.read(ImageBuffer, (NUM_TRAINING_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET);
		ImageStream.close();

	}
	else std::cout << "Couldn't read " + TRAIN_IMAGE_FILE << std::endl;

	std::ifstream LabelStream(TRAIN_LABEL_FILE.c_str(), std::ifstream::binary);
	char * LabelBuffer = new char[NUM_TRAINING_IMAGES + LABEL_OFFSET];

	if (LabelStream)
	{

		LabelStream.read(LabelBuffer, NUM_TRAINING_IMAGES + LABEL_OFFSET);
		LabelStream.close();

	}
	else std::cout << "Couldn't read " + TRAIN_LABEL_FILE << std::endl;

	for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
	{

		TrainingImages[i].Number = LabelBuffer[i + LABEL_OFFSET];
		std::copy(ImageBuffer + IMAGE_OFFSET + (i * IMAGE_WIDTH * IMAGE_HEIGHT), ImageBuffer + IMAGE_OFFSET + ((i + 1) * IMAGE_WIDTH * IMAGE_HEIGHT), TrainingImages[i].ImageData);

	}

	ImageStream.clear();
	LabelStream.clear();

	delete[] ImageBuffer;
	delete[] LabelBuffer;

	// testing images load

	ImageStream.open(TEST_IMAGE_FILE.c_str(), std::ifstream::binary);
	ImageBuffer = new char[(NUM_TESTING_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET];

	if (ImageStream)
	{

		ImageStream.read(ImageBuffer, (NUM_TESTING_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT) + IMAGE_OFFSET);
		ImageStream.close();

	}
	else std::cout << "Couldn't read " + TEST_IMAGE_FILE << std::endl;

	LabelStream.open(TEST_LABEL_FILE.c_str(), std::ifstream::binary);
	LabelBuffer = new char[NUM_TESTING_IMAGES + LABEL_OFFSET];

	if (LabelStream)
	{

		LabelStream.read(LabelBuffer, NUM_TESTING_IMAGES + LABEL_OFFSET);
		LabelStream.close();

	}
	else std::cout << "Couldn't read " + TEST_LABEL_FILE << std::endl;

	for (int i = 0; i < NUM_TESTING_IMAGES; i++)
	{

		TestingImages[i].Number = LabelBuffer[i + LABEL_OFFSET];
		std::copy(ImageBuffer + IMAGE_OFFSET + (i * IMAGE_WIDTH * IMAGE_HEIGHT), ImageBuffer + IMAGE_OFFSET + ((i + 1) * IMAGE_WIDTH * IMAGE_HEIGHT), TestingImages[i].ImageData);

	}

	delete[] ImageBuffer;
	delete[] LabelBuffer;

	Network test = Network(2, 16, 0.1, 1.0);

	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;

	t1 = std::chrono::high_resolution_clock::now();

	GradientDescent(&test, 900); // training

	t2 = std::chrono::high_resolution_clock::now();

	std::vector<double> result = std::vector<double>(10);

	int correct = 0;
	int max;
	float maxvalue;

	std::vector<int> predict_distribution = std::vector<int>(10);

	for (int j = 0; j < NUM_TESTING_IMAGES; j++)
	{

		result = EvaluateNetwork(&test, &TestingImages[j], -1);

		if (j == 0)
		{

			ConsoleOutputImage(&TestingImages[j]);

			for (int i = 0; i < result.size(); i++) std::cout << i << ": " << result[i] << std::endl; // 

		}

		max = 0;
		maxvalue = result[0];

		for (int i = 1; i < 10; i++)
		{

			if (result[i] > maxvalue)
			{

				max = i;
				maxvalue = result[i];

			}

		}

		predict_distribution[max]++;

		if (max == TestingImages[j].Number) correct++;

	}

	std::cout << 100.0 * double(correct) / double(NUM_TRAINING_IMAGES) << "% correct" << std::endl;

	for (int i = 0; i < 10; i++) std::cout << i << ": " << predict_distribution[i] << std::endl;

	std::cout << "Elapsed time: " << std::chrono::duration<double>(t2 - t1).count() << " sec" << std::endl;

	return 0;

}