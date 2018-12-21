#include "funset.hpp"
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

namespace {

bool isFileExists(const std::string &filename)
{
	std::ifstream fhandle(filename.c_str());
	return fhandle.good();
}

bool check_datafiles(const std::vector<std::string> &data_files)
{
	for (size_t index = 0; index < data_files.size(); index++) {
		if (!(isFileExists(data_files[index]))) {
			LG << "Error: File does not exist: " << data_files[index];
			return false;
		}
	}
	return true;
}

bool setDataIter(mxnet::cpp::MXDataIter *iter, std::string useType, const std::vector<std::string> &data_files, int batch_size)
{
	if (!check_datafiles(data_files))
		return false;

	iter->SetParam("batch_size", batch_size);
	iter->SetParam("shuffle", 1);
	iter->SetParam("flat", 1);

	if (useType == "Train") {
		iter->SetParam("image", data_files[0]);
		iter->SetParam("label", data_files[1]);
	} else if (useType == "Label") {
		iter->SetParam("image", data_files[2]);
		iter->SetParam("label", data_files[3]);
	}

	iter->CreateDataIter();
	return true;
}

} // namespace

////////////////////////////// mnist ////////////////////////
/* reference: 
	https://mxnet.incubator.apache.org/tutorials/c%2B%2B/basics.html
	mxnet_source/cpp-package/example/mlp_cpu.cpp
*/
namespace {

mxnet::cpp::Symbol mlp(const std::vector<int> &layers)
{
	auto x = mxnet::cpp::Symbol::Variable("X");
	auto label = mxnet::cpp::Symbol::Variable("label");

	std::vector<mxnet::cpp::Symbol> weights(layers.size());
	std::vector<mxnet::cpp::Symbol> biases(layers.size());
	std::vector<mxnet::cpp::Symbol> outputs(layers.size());

	for (size_t i = 0; i < layers.size(); ++i) {
		weights[i] = mxnet::cpp::Symbol::Variable("w" + std::to_string(i));
		biases[i] = mxnet::cpp::Symbol::Variable("b" + std::to_string(i));
		mxnet::cpp::Symbol fc = mxnet::cpp::FullyConnected(i == 0 ? x : outputs[i - 1], weights[i], biases[i], layers[i]);
		outputs[i] = i == layers.size() - 1 ? fc : mxnet::cpp::Activation(fc, mxnet::cpp::ActivationActType::kRelu);
	}

	return mxnet::cpp::SoftmaxOutput(outputs.back(), label);
}

} // namespace

int test_mnist_train()
{
	const int image_size = 28;
	const std::vector<int> layers{ 128, 64, 10 };
	const int batch_size = 100;
	const int max_epoch = 10;
	const float learning_rate = 0.1;
	const float weight_decay = 1e-2;

#ifdef _MSC_VER
	std::vector<std::string> data_files = { "E:/GitCode/MXNet_Test/data/mnist/train-images.idx3-ubyte",
						"E:/GitCode/MXNet_Test/data/mnist/train-labels.idx1-ubyte",
						"E:/GitCode/MXNet_Test/data/mnist/t10k-images.idx3-ubyte",
						"E:/GitCode/MXNet_Test/data/mnist/t10k-labels.idx1-ubyte"};
#else
	std::vector<std::string> data_files = { "data/mnist/train-images.idx3-ubyte",
						"data/mnist/train-labels.idx1-ubyte",
						"data/mnist/t10k-images.idx3-ubyte",
						"data/mnist/t10k-labels.idx1-ubyte"};

#endif

	auto train_iter = mxnet::cpp::MXDataIter("MNISTIter");
	setDataIter(&train_iter, "Train", data_files, batch_size);

	auto val_iter = mxnet::cpp::MXDataIter("MNISTIter");
	setDataIter(&val_iter, "Label", data_files, batch_size);

	auto net = mlp(layers);

	mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();  // Use CPU for training

	std::map<std::string, mxnet::cpp::NDArray> args;
	args["X"] = mxnet::cpp::NDArray(mxnet::cpp::Shape(batch_size, image_size*image_size), ctx);
	args["label"] = mxnet::cpp::NDArray(mxnet::cpp::Shape(batch_size), ctx);
	// Let MXNet infer shapes other parameters such as weights
	net.InferArgsMap(ctx, &args, args);

	// Initialize all parameters with uniform distribution U(-0.01, 0.01)
	auto initializer = mxnet::cpp::Uniform(0.01);
	for (auto& arg : args) {
		// arg.first is parameter name, and arg.second is the value
		initializer(arg.first, &arg.second);
	}

	// Create sgd optimizer
	mxnet::cpp::Optimizer* opt = mxnet::cpp::OptimizerRegistry::Find("sgd");
	opt->SetParam("rescale_grad", 1.0 / batch_size)->SetParam("lr", learning_rate)->SetParam("wd", weight_decay);

	// Create executor by binding parameters to the model
	auto *exec = net.SimpleBind(ctx, args);
	auto arg_names = net.ListArguments();

	// Start training
	for (int iter = 0; iter < max_epoch; ++iter) {
		int samples = 0;
		train_iter.Reset();

		auto tic = std::chrono::system_clock::now();
		while (train_iter.Next()) {
			samples += batch_size;
			auto data_batch = train_iter.GetDataBatch();
			// Set data and label
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);

			// Compute gradients
			exec->Forward(true);
			exec->Backward();
			// Update parameters
			for (size_t i = 0; i < arg_names.size(); ++i) {
				if (arg_names[i] == "X" || arg_names[i] == "label") continue;
				opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
		}
		auto toc = std::chrono::system_clock::now();

		mxnet::cpp::Accuracy acc;
		val_iter.Reset();
		while (val_iter.Next()) {
			auto data_batch = val_iter.GetDataBatch();
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);
			// Forward pass is enough as no gradient is needed when evaluating
			exec->Forward(false);
			acc.Update(data_batch.label, exec->outputs[0]);
		}
		float duration = std::chrono::duration_cast<std::chrono::milliseconds>
			(toc - tic).count() / 1000.0;
		LG << "Epoch: " << iter << " " << samples / duration << " samples/sec Accuracy: " << acc.Get();
	}

#ifdef _MSC_VER
	std::string json_file{ "E:/GitCode/MXNet_Test/data/mnist.json" };
	std::string param_file{"E:/GitCode/MXNet_Test/data/mnist.params"};
#else
	std::string json_file{ "data/mnist.json" };
	std::string param_file{"data/mnist.params"};
#endif
	net.Save(json_file);
	mxnet::cpp::NDArray::Save(param_file, exec->arg_arrays);

	delete exec;
	MXNotifyShutdown();

	return 0;
}

int test_mnist_predict()
{
#ifdef _MSC_VER
	std::string json_file{ "E:/GitCode/MXNet_Test/data/mnist.json" };
	std::string param_file{ "E:/GitCode/MXNet_Test/data/mnist.params" };
#else
	std::string json_file{ "E:/GitCode/MXNet_Test/data/mnist.json" };
	std::string param_file{ "E:/GitCode/MXNet_Test/data/mnist.params" };
#endif

	mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();  // Use CPU for predict

	mxnet::cpp::Symbol net = mxnet::cpp::Symbol::Load(json_file);
	std::vector<mxnet::cpp::NDArray> array_list;
	mxnet::cpp::NDArray::Load(param_file, &array_list);
	fprintf(stdout, "array size: %d\n", array_list.size());

	const std::map<std::string, mxnet::cpp::NDArray> args_map;
	//args_map["X"] = 

	return 0;
}
