#include "mshadow_test.hpp"
#include <iostream>
#include <cmath>
#include "mshadow/tensor.h"

// reference: mshadow source code: mshadow/guide

int test_mshadow_1()
{
	// intialize tensor engine before using tensor operation
	mshadow::InitTensorEngine<mshadow::cpu>();

	// assume we have a float space
	float data[20];
	// create a 2 x 5 x 2 tensor, from existing space
	mshadow::Tensor<mshadow::cpu, 3> ts(data, mshadow::Shape3(2, 5, 2));
	// take first subscript of the tensor
	mshadow::Tensor<mshadow::cpu, 2> mat = ts[0];
	// Tensor object is only a handle, assignment means they have same data content
	// we can specify content type of a Tensor, if not specified, it is float bydefault
	mshadow::Tensor<mshadow::cpu, 2, float> mat2 = mat;
	mat = mshadow::Tensor<mshadow::cpu, 1>(data, mshadow::Shape1(10)).FlatTo2D();

	// shaape of matrix, note size order is same as numpy
	fprintf(stdout, "%u X %u matrix\n", mat.size(0), mat.size(1));

	// initialize all element to zero
	mat = 0.0f;
	// assign some values
	mat[0][1] = 1.0f; mat[1][0] = 2.0f;
	// elementwise operations
	mat += (mat + 10.0f) / 10.0f + 2.0f;

	// print out matrix, note: mat2 and mat1 are handles(pointers)
	for (mshadow::index_t i = 0; i < mat.size(0); ++i) {
		for (mshadow::index_t j = 0; j < mat.size(1); ++j) {
			fprintf(stdout, "%.2f ", mat2[i][j]);
		}
		fprintf(stdout, "\n");
	}

	mshadow::TensorContainer<mshadow::cpu, 2> lhs(mshadow::Shape2(2, 3)), rhs(mshadow::Shape2(2, 3)), ret(mshadow::Shape2(2, 2));
	lhs = 1.0;
	rhs = 1.0;
	ret = mshadow::expr::implicit_dot(lhs, rhs.T());
	mshadow::VectorDot(ret[0].Slice(0, 1), lhs[0], rhs[0]);
	fprintf(stdout, "vdot=%f\n", ret[0][0]);
	int cnt = 0;
	for (mshadow::index_t i = 0; i < ret.size(0); ++i) {
		for (mshadow::index_t j = 0; j < ret.size(1); ++j) {
			fprintf(stdout, "%.2f ", ret[i][j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	for (mshadow::index_t i = 0; i < lhs.size(0); ++i) {
		for (mshadow::index_t j = 0; j < lhs.size(1); ++j) {
			lhs[i][j] = cnt++;
			fprintf(stdout, "%.2f ", lhs[i][j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	mshadow::TensorContainer<mshadow::cpu, 1> index(mshadow::Shape1(2)), choosed(mshadow::Shape1(2));
	index[0] = 1; index[1] = 2;
	choosed = mshadow::expr::mat_choose_row_element(lhs, index);
	for (mshadow::index_t i = 0; i < choosed.size(0); ++i) {
		fprintf(stdout, "%.2f ", choosed[i]);
	}
	fprintf(stdout, "\n");

	mshadow::TensorContainer<mshadow::cpu, 2> recover_lhs(mshadow::Shape2(2, 3)), small_mat(mshadow::Shape2(2, 3));
	small_mat = -100.0f;
	recover_lhs = mshadow::expr::mat_fill_row_element(small_mat, choosed, index);
	for (mshadow::index_t i = 0; i < recover_lhs.size(0); ++i) {
		for (mshadow::index_t j = 0; j < recover_lhs.size(1); ++j) {
			fprintf(stdout, "%.2f ", recover_lhs[i][j] - lhs[i][j]);
		}
	}
	fprintf(stdout, "\n");

	rhs = mshadow::expr::one_hot_encode(index, 3);

	for (mshadow::index_t i = 0; i < lhs.size(0); ++i) {
		for (mshadow::index_t j = 0; j < lhs.size(1); ++j) {
			fprintf(stdout, "%.2f ", rhs[i][j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
	mshadow::TensorContainer<mshadow::cpu, 1> idx(mshadow::Shape1(3));
	idx[0] = 8;
	idx[1] = 0;
	idx[2] = 1;

	mshadow::TensorContainer<mshadow::cpu, 2> weight(mshadow::Shape2(10, 5));
	mshadow::TensorContainer<mshadow::cpu, 2> embed(mshadow::Shape2(3, 5));

	for (mshadow::index_t i = 0; i < weight.size(0); ++i) {
		for (mshadow::index_t j = 0; j < weight.size(1); ++j) {
			weight[i][j] = i;
		}
	}
	embed = mshadow::expr::take(idx, weight);
	for (mshadow::index_t i = 0; i < embed.size(0); ++i) {
		for (mshadow::index_t j = 0; j < embed.size(1); ++j) {
			fprintf(stdout, "%.2f ", embed[i][j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n\n");
	weight = mshadow::expr::take_grad(idx, embed, 10);
	for (mshadow::index_t i = 0; i < weight.size(0); ++i) {
		for (mshadow::index_t j = 0; j < weight.size(1); ++j) {
			fprintf(stdout, "%.2f ", weight[i][j]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "upsampling\n");

#ifdef small
#undef small
#endif

	mshadow::TensorContainer<mshadow::cpu, 2> small(mshadow::Shape2(2, 2));
	small[0][0] = 1.0f;
	small[0][1] = 2.0f;
	small[1][0] = 3.0f;
	small[1][1] = 4.0f;
	mshadow::TensorContainer<mshadow::cpu, 2> large(mshadow::Shape2(6, 6));
	large = mshadow::expr::upsampling_nearest(small, 3);
	for (mshadow::index_t i = 0; i < large.size(0); ++i) {
		for (mshadow::index_t j = 0; j < large.size(1); ++j) {
			fprintf(stdout, "%.2f ", large[i][j]);
		}
		fprintf(stdout, "\n");
	}
	small = mshadow::expr::pool<mshadow::red::sum>(large, small.shape_, 3, 3, 3, 3);
	// shutdown tensor enigne after usage
	for (mshadow::index_t i = 0; i < small.size(0); ++i) {
		for (mshadow::index_t j = 0; j < small.size(1); ++j) {
			fprintf(stdout, "%.2f ", small[i][j]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "mask\n");
	mshadow::TensorContainer<mshadow::cpu, 2> mask_data(mshadow::Shape2(6, 8));
	mshadow::TensorContainer<mshadow::cpu, 2> mask_out(mshadow::Shape2(6, 8));
	mshadow::TensorContainer<mshadow::cpu, 1> mask_src(mshadow::Shape1(6));

	mask_data = 1.0f;
	for (int i = 0; i < 6; ++i) {
		mask_src[i] = static_cast<float>(i);
	}
	mask_out = mshadow::expr::mask(mask_src, mask_data);
	for (mshadow::index_t i = 0; i < mask_out.size(0); ++i) {
		for (mshadow::index_t j = 0; j < mask_out.size(1); ++j) {
			fprintf(stdout, "%.2f ", mask_out[i][j]);
		}
		fprintf(stdout, "\n");
	}

	mshadow::ShutdownTensorEngine<mshadow::cpu>();

	return 0;
}

// user defined unary operator addone
struct addone {
	// map can be template function
	template<typename DType>
	MSHADOW_XINLINE static DType Map(DType a) {
		return  a + static_cast<DType>(1);
	}
};
// user defined binary operator max of two
struct maxoftwo {
	// map can also be normal functions,
	// however, this can only be applied to float tensor
	MSHADOW_XINLINE static float Map(float a, float b) {
		if (a > b) return a;
		else return b;
	}
};

int test_mshadow_2()
{
	// intialize tensor engine before using tensor operation, needed for CuBLAS
	mshadow::InitTensorEngine<mshadow::cpu>();
	// take first subscript of the tensor
	mshadow::Stream<mshadow::cpu> *stream_ = mshadow::NewStream<mshadow::cpu>(0);
	mshadow::Tensor<mshadow::cpu, 2, float> mat = mshadow::NewTensor<mshadow::cpu>(mshadow::Shape2(2, 3), 0.0f, stream_);
	mshadow::Tensor<mshadow::cpu, 2, float> mat2 = mshadow::NewTensor<mshadow::cpu>(mshadow::Shape2(2, 3), 0.0f, stream_);

	mat[0][0] = -2.0f;
	mat = mshadow::expr::F<maxoftwo>(mshadow::expr::F<addone>(mat) + 0.5f, mat2);

	for (mshadow::index_t i = 0; i < mat.size(0); ++i) {
		for (mshadow::index_t j = 0; j < mat.size(1); ++j) {
			fprintf(stdout, "%.2f ", mat[i][j]);
		}
		fprintf(stdout, "\n");
	}

	mshadow::FreeSpace(&mat); mshadow::FreeSpace(&mat2);
	mshadow::DeleteStream(stream_);
	// shutdown tensor enigne after usage
	mshadow::ShutdownTensorEngine<mshadow::cpu>();

	return 0;
}
