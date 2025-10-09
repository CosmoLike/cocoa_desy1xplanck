#define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_HDF5
#include <carma.h>
#include <armadillo>
#include <map>

#ifndef __COSMOLIKE_INTERFACE_HPP
#define __COSMOLIKE_INTERFACE_HPP

// --- Auxiliary Code ---
namespace interface_mpp_aux
{

class RandomNumber
{
// Singleton Class that holds a random number generator

public:
  static RandomNumber& get_instance()
  {
    static RandomNumber instance;
		return instance;
  }
  ~RandomNumber() = default;

  double get()
  {
    return dist_(mt_);
  }

protected:
  std::random_device rd_;
  std::mt19937 mt_;
  std::uniform_real_distribution<double> dist_;
private:
  RandomNumber() :
    rd_(),
    mt_(rd_()),
    dist_(0.0, 1.0) {
	};
  RandomNumber(RandomNumber const&) = delete;
};

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

class RealData
{
// Singleton Class that holds a data vector, covariance, maps...

public:
  static RealData& get_instance()
  {
    static RealData instance;
    return instance;
  }

  ~RealData() = default;

  void set_data(std::string DATA);

  void set_mask(std::string MASK);

  void set_inv_cov(std::string COV);

  void set_PMmarg(std::string U_PMmarg_file);

  void set_cmb_binning_mat(std::string BINMAT);

  void set_cmb_theory_offset(std::string OFFSET);

  int get_ndim() const;

  int get_nreduced_dim() const;

  int get_index_reduced_dim(const int ci) const;

  arma::Col<int> get_mask() const;

  int get_mask(const int ci) const;

  arma::Col<double> get_data_masked() const;

  double get_data_masked(const int ci) const;

  arma::Col<double> get_data_masked_reduced_dim() const;

  double get_data_masked_reduced_dim(const int ci) const;

  arma::Mat<double> get_covariance_masked() const;

  arma::Mat<double> get_covariance_masked_reduced_dim() const;

  arma::Mat<double> get_inverse_covariance_masked() const;

  double get_inverse_covariance_masked(const int ci, const int cj) const;

  arma::Mat<double> get_inverse_covariance_masked_reduced_dim() const;

  double get_inverse_covariance_masked_reduced_dim(const int ci, const int cj) const;

  double get_binning_matrix_with_correction(const int ci, const int cj) const;

  arma::Mat<double> get_binning_matrix_with_correction() const;

  double get_cmb_theory_offset(const int ci) const;

  arma::Mat<double> get_cmb_theory_offset() const;

  double get_chi2(std::vector<double> datavector) const;

  bool is_mask_set() const;

  bool is_data_set() const;

  bool is_inv_cov_set() const;

  bool is_cmb_binmat_set() const;

  bool is_cmb_offset_set() const;

  arma::Col<double> get_expand_dim_from_masked_reduced_dim(
    arma::Col<double> reduced_dim_vector) const;

private:
  bool is_mask_set_ = false;
  bool is_data_set_ = false;
  bool is_inv_cov_set_ = false;
  bool is_cmb_binmat_set_ = false;
  bool is_cmb_offset_set_ = false;

  int ndata_;
  int ndata_masked_; // for baryon project, reduced dim
  int nbp_;
  int ncl_;

  std::string mask_filename_;
  std::string cov_filename_;
  std::string data_filename_;
  std::string binmat_filename_; // DESxPlanck
  std::string offset_filename_; // DESxPlanck

  arma::Col<int> index_reduced_dim_;
  arma::Col<int> mask_;

  arma::Col<double> data_masked_;
  arma::Col<double> data_masked_reduced_dim_; // for baryon project, reduced dim

  arma::Mat<double> cov_masked_;
  arma::Mat<double> cov_masked_reduced_dim_; // for baryon project, reduced dim
  arma::Mat<double> inv_cov_masked_;
  arma::Mat<double> inv_cov_masked_reduced_dim_;

  arma::Mat<double> cmb_binning_matrix_with_correction_; // DESxPlanck, see Eqn 35 Planck 2018 VIII
  arma::Col<double> cmb_theory_offset_;                  // DESxPlanck, see Eqn 35 Planck 2018 VIII

  RealData() = default;
  RealData(RealData const&) = delete;
};

arma::Mat<double> read_table(const std::string file_name);

std::vector<double> convert_arma_col_to_stl_vector(arma::Col<double> in);

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp = 100)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
}

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

class PointMass
{
// Singleton Class that Evaluate Point Mass Marginalization

public:
  static PointMass& get_instance()
  {
    static PointMass instance;
    return instance;
  }
  ~PointMass() = default;

  void set_pm_vector(std::vector<double> pm);

  std::vector<double> get_pm_vector() const;

  double get_pm(const int zl, const int zs, const double theta) const;

  double get_pm_exact(const int zl, const int zs, const double theta) const;

private:
  std::vector<double> pm_;

  PointMass() = default;
  PointMass(PointMass const&) = delete;
};

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

class BaryonScenario
{
// Singleton Class that map Baryon Scenario (integer to name)

public:
  static BaryonScenario& get_instance()
  {
    static BaryonScenario instance;
    return instance;
  }
  ~BaryonScenario() = default;

  int nscenarios() const;

  void set_scenarios(std::string scenarios);

  std::string get_scenario(const int i) const;

private:
  int nscenarios_;
  std::map<int, std::string> scenarios_;

  BaryonScenario() = default;
  BaryonScenario(BaryonScenario const&) = delete;
};

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

}  // namespace interface_mpp_aux
#endif // HEADER GUARD
