#include "dmlc_test.hpp"
#include <iostream>
#include <cstdio>
#include <functional>
#include <dmlc/parameter.h>
#include <dmlc/registry.h>

// reference: dmlc-core/example and dmlc-core/test

struct MyParam : public dmlc::Parameter<MyParam> {
	float learning_rate;
	int num_hidden;
	int activation;
	std::string name;
	// declare parameters in header file
	DMLC_DECLARE_PARAMETER(MyParam) {
		DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000)
			.describe("Number of hidden unit in the fully connected layer.");
		DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f)
			.describe("Learning rate of SGD optimization.");
		DMLC_DECLARE_FIELD(activation).add_enum("relu", 1).add_enum("sigmoid", 2)
			.describe("Activation function type.");
		DMLC_DECLARE_FIELD(name).set_default("mnet")
			.describe("Name of the net.");

		// user can also set nhidden besides num_hidden
		DMLC_DECLARE_ALIAS(num_hidden, nhidden);
		DMLC_DECLARE_ALIAS(activation, act);
	}
};

// register it in cc file
DMLC_REGISTER_PARAMETER(MyParam);

int test_dmlc_parameter()
{
	int argc = 4;
	char* argv[4] = {
#ifdef _DEBUG
		"E:/GitCode/MXNet_Test/lib/dbg/x64/ThirdPartyLibrary_Test.exe",
#else
		"E:/GitCode/MXNet_Test/lib/rel/x64/ThirdPartyLibrary_Test.exe",
#endif
		"num_hidden=100",
		"name=aaa",
		"activation=relu"
	};

	MyParam param;
	std::map<std::string, std::string> kwargs;
	for (int i = 0; i < argc; ++i) {
		char name[256], val[256];
		if (sscanf(argv[i], "%[^=]=%[^\n]", name, val) == 2) {
			kwargs[name] = val;
		}
	}
	fprintf(stdout, "Docstring\n---------\n%s", MyParam::__DOC__().c_str());

	fprintf(stdout, "start to set parameters ...\n");
	param.Init(kwargs);
	fprintf(stdout, "-----\n");
	fprintf(stdout, "param.num_hidden=%d\n", param.num_hidden);
	fprintf(stdout, "param.learning_rate=%f\n", param.learning_rate);
	fprintf(stdout, "param.name=%s\n", param.name.c_str());
	fprintf(stdout, "param.activation=%d\n", param.activation);

	return 0;
}

namespace tree {
	struct Tree {
		virtual void Print() = 0;
		virtual ~Tree() {}
	};

	struct BinaryTree : public Tree {
		virtual void Print() {
			printf("I am binary tree\n");
		}
	};

	struct AVLTree : public Tree {
		virtual void Print() {
			printf("I am AVL tree\n");
		}
	};
	// registry to get the trees
	struct TreeFactory
		: public dmlc::FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
	};

#define REGISTER_TREE(Name)                                             \
  DMLC_REGISTRY_REGISTER(::tree::TreeFactory, TreeFactory, Name)        \
  .set_body([]() { return new Name(); } )

	DMLC_REGISTRY_FILE_TAG(my_tree);

}  // namespace tree

// usually this sits on a seperate file
namespace dmlc {
	DMLC_REGISTRY_ENABLE(tree::TreeFactory);
}

namespace tree {
	// Register the trees, can be in seperate files
	REGISTER_TREE(BinaryTree)
		.describe("This is a binary tree.");

	REGISTER_TREE(AVLTree);

	DMLC_REGISTRY_LINK_TAG(my_tree);
}

int test_dmlc_registry()
{
	// construct a binary tree
	tree::Tree *binary = dmlc::Registry<tree::TreeFactory>::Find("BinaryTree")->body();
	binary->Print();
	// construct a binary tree
	tree::Tree *avl = dmlc::Registry<tree::TreeFactory>::Find("AVLTree")->body();
	avl->Print();

	delete binary;
	delete avl;

	return 0;
}