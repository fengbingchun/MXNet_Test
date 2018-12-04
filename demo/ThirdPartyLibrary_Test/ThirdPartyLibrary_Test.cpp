#include <iostream>
#include "openblas_test.hpp"
#include "mshadow_test.hpp"
#include "dmlc_test.hpp"

int main()
{
	int ret = test_dmlc_registry();

	if (ret == 0) fprintf(stdout, "====== test success ======\n");
	else fprintf(stderr, "###### test fail ######\n");
}
