#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_mnist_predict();
	if (ret == 0) fprintf(stdout, "====== test success ======\n");
	else fprintf(stderr, "###### test fail ######\n");
}
