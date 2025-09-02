//
//  epiSize_sim.cpp
//  
//
//  Created by Brandon Simony on 3/13/24.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cassert>
#include <iomanip>
#include "gsl/gsl_randist.h"


//-------------------------//
// RV Generators using GSL //
//-------------------------//

//For initiating and seeding gsl random number generator.
class Gsl_rng_wrapper
{
    gsl_rng* r;
    public:
        Gsl_rng_wrapper()
        {
            std::random_device rng_dev; //To generate a safe seed.
            long seed = time(NULL)*rng_dev();
            const gsl_rng_type* rng_type = gsl_rng_default;
            r = gsl_rng_alloc(rng_type);
            gsl_rng_set(r, seed);
        }
        ~Gsl_rng_wrapper() { gsl_rng_free(r); }
        gsl_rng* get_r() { return r; }
};

// Uniform RV using gsl.
double draw_uniform_gsl(double lo, double hi)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    return gsl_ran_flat(r, lo, hi);
}

// Discrete RV using gsl.
int draw_discrete_gsl(const std::vector<double>& probs)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r = nullptr;
    if (r == nullptr) { r = rng_w.get_r(); }
    gsl_ran_discrete_t* dist = gsl_ran_discrete_preproc(probs.size(), probs.data());
    int result = static_cast<int>(gsl_ran_discrete(r, dist));
    gsl_ran_discrete_free(dist);

    return result;
}

//Binomial RV using gsl.
int draw_binom_gsl(int N, double prob)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    return gsl_ran_binomial(r, prob, N);
}

//Multinomial RV using gsl.
std::vector<int> draw_multinom_gsl(int N, const std::vector<double>& prob)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    size_t K = prob.size();
    std::vector<unsigned int> counts(K, 0);
    gsl_ran_multinomial(r, K, N, prob.data(), counts.data());
    std::vector<int> result(counts.begin(), counts.end());
    return result;
}

//Exponential RV using gsl. parameterized by a mean
double draw_exp_gsl(double mu)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    return gsl_ran_exponential(r, mu);
}

//Gamma RV using gsl.
double draw_gamma_gsl(double shape, double scale)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    return gsl_ran_gamma(r, shape, scale);
}

//Poisson RV using gsl
int draw_poisson_gsl(double rate)
{
    static Gsl_rng_wrapper rng_w;
    static gsl_rng* r;
    if(r == nullptr){ r = rng_w.get_r(); }
    return gsl_ran_poisson(r, rate);
}


//----------------------//
// Core model structure //
//----------------------//

// struct to hold parameter value inputs
struct Parameters
{
    int n_drugs = 1;
    int n_classes = 2;
    
    std::vector<int> N_0; // initial population vector
    
    std::vector<double> b_vec; // type-specific birth rate
    std::vector<double> d_vec; // type-specific death rate
    std::vector<double> mu_vec; // per-site mutation probability
    std::vector<double> k_vec; // drug efficacy factor
    
    int T_max = 10; // number of generations
    int N_max = 1e9; // critical population size
    int gen_step = 1; // steps between output generations
    int verbose = 0; // debugging statements
    int data_out = 0; // specify model type. 0: console; 1: to file
    std::string batchname = "outSize_distn"; // output file name tag
    std::string par_file = "pars.txt"; // output file name tag
    
    std::vector<std::string> conf_v;

    //Constructor either reads parameters from standard input (if no argument is given),
    //or from file (argument 1(.
    Parameters(int argc, char* argv[])
    {
        conf_v.reserve(200);
        std::stringstream buffer;
        if(argc == 3) //Config file
        {
            std::ifstream f(argv[2]);
            if(f.is_open())
            {
                buffer << f.rdbuf();
            }
            else
            {
                std::cout << "Failed to read config file \"" << argv[2] << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            for(int i=2; i<argc; ++i)
            {
                buffer << argv[i] << std::endl;
            }
        }
        while(!buffer.eof())
        { // until end of the stream
            std::string line = "";
            std::stringstream line_ss;
            // First get a whole line from the config file
            std::getline(buffer, line);
            // Put that in a stringstream (required for getline) and use getline again
            // to only read up until the comment character (delimiter).
            line_ss << line;
            std::getline(line_ss, line, '#');
            // If there is whitespace between the value of interest and the comment character
            // this will be read as part of the value. Therefore strip it of whitespace first.
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line.size() != 0)
            {
                conf_v.push_back(line);
            }
        }

        // must be read in before evaluating length since the number of classes (and thus parameters) is variable based on the number of variable sites / drugs
        
        size_t conf_length = 8; //number of model input parameters
        if (conf_v.size()!=conf_length)
        {
            std::cout << "Expected configuration file with " << conf_length << " options, loaded file with "
                      << conf_v.size() << " lines." << std::endl;
            exit(EXIT_FAILURE);
        }

        n_drugs = std::stoi(conf_v[0]); // number of drugs::resistance sites
        n_classes = pow(2, n_drugs);
        
        T_max = std::stoi(conf_v[1]); // simulation length
        N_max = std::stoi(conf_v[2]); // critical population size
        gen_step = std::stoi(conf_v[3]); // select step between output generations
    
        // simulation output
        verbose = std::stoi(conf_v[4]); // verbose statements (always written to batchname_err.txt file)
        data_out = std::stoi(conf_v[5]); // output indicator
        batchname = conf_v[6]; // output file name tag
        par_file = conf_v[7]; // "path/f_name.txt" for vectorized parameters
        
        //-------------------------------//
        // read in vectorized parameters //
        //-------------------------------//
        
        std::string file_path = "./simulation_files/" + par_file + ".txt";
        std::ifstream parms(file_path);
        if (parms.is_open()) {
            std::string line;
            int lineIndex = 0;

            while (std::getline(parms, line)) {
                if (line.empty()) continue; // skip blank lines

                std::stringstream iss(line);
                std::string tmp;

                switch (lineIndex) {
                    case 0: // N_j
                        while (std::getline(iss, tmp, ',')) {
                            N_0.push_back(std::stoi(tmp));
                        }
                        break;
                    case 1: // b
                        while (std::getline(iss, tmp, ',')) {
                            b_vec.push_back(std::stod(tmp));
                        }
                        break;
                    case 2: // d
                        while (std::getline(iss, tmp, ',')) {
                            d_vec.push_back(std::stod(tmp));
                        }
                        break;
                    case 3: // mu
                        while (std::getline(iss, tmp, ',')) {
                            mu_vec.push_back(std::stod(tmp));
                        }
                        break;
                    case 4: // k
                        while (std::getline(iss, tmp, ',')) {
                            k_vec.push_back(std::stod(tmp));
                        }
                        break;
                    default:
                        std::cerr << "Warning: extra line in parameter file ignored: "
                                  << line << std::endl;
                        break;
                }

                lineIndex++;
            }

        } else {
            std::cout << "Failed to open file: " << file_path << std::endl;
            exit(EXIT_FAILURE);
        }
        
        parms.close();
        
        //-------------------------//
        // validate vector lengths //
        //-------------------------//
        
        // helper function for checks and error message
        auto check_size = [](const auto& vec, size_t expected,
                             const std::string& name, const std::string& ref,
                             std::ostringstream& err) {
            if (vec.size() != expected) {
                err << "Size mismatch: " << name << " (" << vec.size()
                    << ") vs " << ref << " (" << expected << ")\n";
            }
        };

        std::ostringstream err;
        
        // check for size mismatch
        check_size(k_vec, n_drugs, "k_vec", "n_drugs", err);
        check_size(mu_vec, n_drugs, "mu_vec", "n_drugs", err);
        check_size(N_0, n_classes, "N_0", "n_classes", err);
        check_size(b_vec, n_classes, "b_vec", "n_classes", err);
        check_size(d_vec, n_classes, "d_vec", "n_classes", err);
        

        // print all mismatch errors and terminate
        if (!err.str().empty()) {
            throw std::runtime_error(err.str());
        }

    }
};


// structure to contain population size of all classes at each generation
struct treeData {
    
    std::vector<std::vector<int>> popData; // counts[gen][class]
    std::vector<int> N; // counts sum over all classes
    int extinct = 0; // indicator if simulation went extinct
    int crit = 0; // indicator if simulation became critical (i.e., N>p.N_max)
    int T_extinct = 0; // time of extinction, if applicable
    treeData(int n_gen, int n_classes) {
        popData.assign(n_gen, std::vector<int>(n_classes, 0)); // initialize counts to 0
        N.assign(n_gen, 0);
    }
    
    // Accessor (clearer than raw counts[g][c])
    int& at(int gen, int cls) { return popData[gen][cls]; }
    
};


// simulation code for discrete branching process
// this corresponds to the discrete time generating function
treeData branchingSim_byGen( std::ofstream &errOut,
                             const Parameters& p,
                             const std::vector<std::vector<double>>& mu_prob,
                             const std::vector<double>& birth_probs ){
    
    treeData data(p.T_max+1, p.n_classes);

    // initialize counts at T = 0
    std::vector<int> current_gen = p.N_0;
    std::vector<int> next_gen(current_gen.size(), 0);
    data.popData[0] = current_gen;
    data.N[0] = std::accumulate(current_gen.begin(), current_gen.end(), 0); // total population size
    
    for (int T = 0; T < p.T_max; T++) {
        
        // Loop over types
        for (int type = 0; type < p.n_classes; ++type) {
            if (current_gen[type] == 0) continue;

            // number of individuals that reproduce this generation
            int n_reproduce = draw_binom_gsl(current_gen[type], birth_probs[type]);
            std::vector<int> offspring_vector = draw_multinom_gsl(n_reproduce, std::vector<double>(mu_prob[type].begin(), mu_prob[type].end()) );
            
            // Add offspring to next generation
            for (int j = 0; j < offspring_vector.size(); ++j) {
                next_gen[j] += offspring_vector[j];
            }

            // Add the parents back to their own type
            next_gen[type] += n_reproduce;
            
        }

        // stop early if negative value occurs, most likely due to large population overflow
        // abort before new (negative) values are added and replaced.
        if ( std::any_of(next_gen.begin(), next_gen.end(), [](double x){ return x < 0.0; }) ) {
            errOut << "Suspected overflow detected due to negative class population. Aborting replicate simulation at generation T = " << T+1 << std::endl;
            errOut << "Population size before error: " << data.N[T] << std::endl;
            
            errOut << "Printing current population values: " << std::endl;
            for(int i = 0; i < current_gen.size(); i++){
                errOut << current_gen[i] << "; ";
            }
            errOut << std::endl;
            
            errOut << "Printing next generation population values: " << std::endl;
            for(int i = 0; i < next_gen.size(); i++){
                errOut << next_gen[i] << "; ";
            }
            errOut << std::endl;
            
            data.N.resize(T+1); // Resize N to only hold the valid population steps
            break;
        }
        
        // store current counts
        data.popData[T+1]  = next_gen;
        data.N[T+1] = std::accumulate(next_gen.begin(), next_gen.end(), 0); // total population size
        
        // early stop if extinct (i.e., next gen. all zero), save extinction values
        if ( data.N[T+1] == 0 ) {
            data.extinct = 1; data.T_extinct = T+1;
            data.N.resize(T+2); // Resize N to remove time points after extinction -- argument is size not max index
            errOut << "Extinction occurred at generation " << T+1 << std::endl;
            break;
        }
        
        // early stop if process exceeds population threshold
        if ( data.N[T+1] >= p.N_max) {
            data.crit = 1;
            data.N.resize(T+2); // Resize N to remove time points after critical point -- argument is size not max index
            errOut << "Critical population of size " << data.N[T+1] << " at generation " << T+1 << std::endl;
            break;
        }
        
        // reset current/future state storage containers
        current_gen = next_gen; // next_gen becomes current_gen to start next iteration
        next_gen.assign(next_gen.size(), 0); // next_gen is empty to start next iteration
       
    }
    
    return(data);
    
}


//------------------------//
// Misc. helper functions //
//------------------------//

// Function to generate interpretable class names
std::vector<std::string> generateClassNames(int D) {
    std::vector<std::string> names;
    int total = 1 << D; // 2^K classes

    for (int mask = 0; mask < total; ++mask) {
        if (mask == 0) {
            names.push_back("WT"); // Wild type (no resistance)
        } else {
            std::string label = "M";
            for (int i = 0; i < D; ++i) {
                if (mask & (1 << i)) {
                    label += std::to_string(i + 1); // e.g., M1, M12
                }
            }
            names.push_back(label);
        }
    }
    return names;
}


// use bit-mapped type to determine number of mutations per individual
int numMutations(int x) {
    int count = 0;
    while (x) {
        count += x & 1;   // add 1 if lowest bit is set
        x >>= 1;          // shift right
    }
    return count;
}


// Function to compute offspring probability matrix
// P_{i,j} = P(type i has 1 type j offspring)
std::vector<std::vector<double>> offspringMatrix(const std::vector<double>& mu) {
    int K = mu.size();         // number of sites
    int numTypes = 1 << K; // 2^K possible genotypes -- bit representation
    
    std::vector<std::vector<double>> offspring_mat(numTypes, std::vector<double>(numTypes, -1.0));

    // For each parent genotype
    for (int parent = 0; parent < numTypes; ++parent) {
        // For each possible child genotype
        for (int child = 0; child < numTypes; ++child) {
            double prob = 1.0;
            // Loop over each site to compute probability of matching parent->child
            for (int site = 0; site < K; ++site) {
                bool parentHas = parent & (1 << site);
                bool childHas  = child  & (1 << site);

                if (parentHas == childHas) {
                    // no mutation at this site
                    prob *= (1.0 - mu[site]);
                } else {
                    // mutation occurred at this site
                    prob *= mu[site];
                }
            }
            offspring_mat[parent][child] = prob;
        }
    }
    return offspring_mat;
}


// Function for birth / death probabilities for each type given treatment
// calculates individual per-type birth probability when treatment is employed
// requires parameterization for alternative functional drug actions
std::vector<double> birthProbs(const Parameters& p){
    
    // declare b and d variables for
    // usage in lambda helper functions
    double b; double d;
    double pmc = 0.01; // hard coded birth rate cost per mutation
    
    //------------------------------------//
    //     helper lambda functions for    //
    // different birth/death augmentation //
    //------------------------------------//
    
    // multiplies death by factor of k_j if susceptible
    auto f = [&b, &d](double k) -> std::vector<double> {
        double b_star = b;
        double d_star = k * d;
        return {b_star, d_star};
    };
    
    // reduction of birth -- WIP
    auto g = [&b, &d](double k) -> std::vector<double> {
        double b_star = b;
        double d_star = d;
        return {b_star, d_star};
    };
    
    // reduction of birth from fitness costs by # mutations -- WIP
    // n_mut = numMutations(cls);
    auto fit_cost = [&b, &d, &pmc](int n_mut) -> std::vector<double> {
        double b_star = b*pow(pmc, n_mut);
        double d_star = d;
        return {b_star, d_star};
    };
    
    // etc...
    
    
    //---------------------//
    // start function body //
    //---------------------//
    
    std::vector<double> birthProbs;
    birthProbs.assign(p.n_classes, 0.0);
    
    for(int cls = 0; cls < p.n_classes; cls++) {
        
        std::vector<double> typej_probs(p.n_drugs, 0.0);
        
        // class-specific birth and death rates
        b = p.b_vec[cls];
        d = p.d_vec[cls];

        for(int drugType = 0; drugType < p.n_drugs; drugType++){
            
            // check for resistance to current drug type
            if( (cls >> drugType) & 1 ){
                // resistant, so no change
                typej_probs[drugType] = b / (b + d);
            }else{
                // susceptible, so apply augmentation
                std::vector<double> new_bd = f(p.k_vec[drugType]);
                typej_probs[drugType] = new_bd[0] / (new_bd[0] + new_bd[1]);
            }
            
        } // end loop over drug types
        
        birthProbs[cls] = *std::min_element(typej_probs.begin(), typej_probs.end()); // keep minimum birth probability -- implies only the strongest drug applies mortality
        
    } // end loop over classes
    
    return birthProbs;
    
}


//-------------------------------------------//
// Function main performs parameter read-in  //
// and manages rreplicate results and output //
//-------------------------------------------//

int main(int argc, char* argv[]){
    
    // initialize parameters read from command line input
    int n_reps = std::stoi(argv[1]); // simulation replicates
    Parameters p(argc, argv);
    
    
    if(p.verbose > 2){ std::cout << std::filesystem::current_path() << std::endl; }
    
    // open error file
    std::string err_fname = "./simulation_files/" + p.batchname + "_err.txt";
    std::ofstream errOut(err_fname);
    if (!errOut.is_open()) {
        std::cerr << "Unable to open error file: " << err_fname << std::endl;
        return EXIT_FAILURE;
    }
    
    // open full output file
    std::string fname = "./simulation_files/" + p.batchname + "_results.txt";
    std::ofstream simResults(fname);
    if (!simResults.is_open()) {
        std::cerr << "Unable to open error file: " << fname << std::endl;
        return EXIT_FAILURE;
    }
    
    // open full output file
    std::string extinct_fname = "./simulation_files/" + p.batchname + "_extinction.txt";
    std::ofstream extinct(extinct_fname);
    if (!extinct.is_open()) {
        std::cerr << "Unable to open error file: " << extinct_fname << std::endl;
        return EXIT_FAILURE;
    }
    
    // open "final step only" output file
    std::string end_fname = "./simulation_files/" + p.batchname + "_end.txt";
    std::ofstream endResults(end_fname);
    if (!endResults.is_open()) {
        std::cerr << "Unable to open error file: " << end_fname << std::endl;
        return EXIT_FAILURE;
    }
    
    // write current runtime to .err file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    errOut << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << std::endl;
    // std::cout << "Wrote to " << err_fname << std::endl;
    
    if(p.verbose > 0){
        
        errOut << "###################################\n"; // end of row
        errOut << "Printing non-vector parameters \n"; // end of row
        errOut << "###################################\n"; // end of row
        errOut << "n_drugs: " << p.n_drugs << "\n";
        errOut << "n_classes: " << p.n_classes << "\n";
        errOut << "T_max: " << p.T_max << "\n";
        errOut << "N_max: " << p.N_max << "\n";
        errOut << "gen_step: " << p.gen_step << "\n";
        errOut << "verbose: " << p.verbose << "\n";
        errOut << "data_out: " << p.data_out << "\n";
        errOut << "batchname: " << p.batchname << "\n";
        errOut << "par_dat: " << p.par_file << "\n" << "\n" << std::endl;

    }
    
    if(p.verbose > 0){
        
        errOut << "#############################\n"; // end of row
        errOut << "Printing vector parameters \n"; // end of row
        errOut << "#############################\n"; // end of row
        
        errOut << "N_0: ";
        for (size_t i = 0; i < p.N_0.size(); ++i) { errOut << p.N_0[i] << "; "; }
        errOut << "\n";
        
        errOut << "b_vec: ";
        for (size_t i = 0; i < p.b_vec.size(); ++i) { errOut << p.b_vec[i] << "; "; }
        errOut << "\n";
        
        errOut << "d_vec: ";
        for (size_t i = 0; i < p.d_vec.size(); ++i) { errOut << p.d_vec[i] << "; "; }
        errOut << "\n";
        
        errOut << "mu_vec: ";
        for (size_t i = 0; i < p.mu_vec.size(); ++i) { errOut << p.mu_vec[i] << "; "; }
        errOut << "\n";
        
        errOut << "k_vec: ";
        for (size_t i = 0; i < p.k_vec.size(); ++i) { errOut << p.k_vec[i] << "; "; }
        errOut << "\n" << "\n" << std::endl;
        
    }
    
    
    //-----------------------------//
    // pre-simulation computations //
    //-----------------------------//
    
    // calculate by-type offspring probability distribution  matrix
    std::vector<std::vector<double>> prob_matrix = offspringMatrix(p.mu_vec);
    
        
    // troubleshoot/debug visual for offspring matrix
    if(p.verbose > 0){
        
        errOut << "#####################################\n"; // end of row
        errOut << "Printing offspring prob. matrix\n"; // end of row
        errOut << "#####################################\n"; // end of row
        for (size_t i = 0; i < prob_matrix.size(); ++i) {
            for (size_t j = 0; j < prob_matrix[i].size(); ++j) {
                errOut << std::setw(10)               // fix column width
                       << std::scientific << std::setprecision(3)        // 4 digits after decimal
                       << prob_matrix[i][j];
                if (j < prob_matrix[i].size() - 1) errOut << " ";
            }
            errOut << "\n";
        }
        errOut << "\n" << std::endl;
    }
    
    // calculate by type birth probability
    std::vector<double> birth_vec = birthProbs( p );
        
    // troubleshoot/debug visual for birth vector
    if(p.verbose > 0){
        
        errOut << "#############################\n"; // end of row
        errOut << "Printing birth prob. vector\n"; // end of row
        errOut << "#############################\n"; // end of row
        for (size_t i = 0; i < birth_vec.size(); ++i) {
            errOut << birth_vec[i] << "; ";
        }
        errOut << "\n" << "\n" << std::endl; // end of row
    }
    //------------------------------------//
    // generate dynamic labels for header //
    //------------------------------------//
    
    std::vector<std::string> class_names = generateClassNames(p.n_drugs); // generate all class names for n_drugs drugs
    if(p.verbose > 0){
        
        errOut << "#######################\n"; // end of row
        errOut << "Printing class names\n"; // end of row
        errOut << "#######################\n"; // end of row
        for (size_t i = 0; i < class_names.size(); ++i) {
            errOut << class_names[i] << "; ";
        }
        errOut << "\n" << "\n" << std::endl; // end of row
    }
    
    //--------------------//
    // extinction metrics //
    //--------------------//
    double ext = 0; double T_ext = 0; double crit = 0;
    extinct << "prob; " << "T_mean;" << "crit" << std::endl;
    
    
    //----------------------------//
    // select model output format //
    //----------------------------//
    
    // troubleshoot/debug visual for birth vector
    if(p.verbose > 0){
        
        errOut << "#######################\n"; // end of row
        errOut << "Running Simulations\n"; // end of row
        errOut << "#######################\n"; // end of row
    }
    
    switch (p.data_out) {
            
        // write to command line
        case 0:
            
            // write header with dynamic labels
            std::cout << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                std::cout << ";" << class_names[cls];
            }
            std::cout << ";N" << "\n";
            
            endResults << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                endResults << ";" << class_names[cls];
            }
            endResults << ";N" << std::endl;
            
            
            // model results by replicate
            for(int i = 0; i < n_reps; i++){
                // run model
                if(p.verbose > 0){ errOut << "Rep. " << i+1 << "\n"; }
                treeData res_out = branchingSim_byGen( errOut, p, prob_matrix, birth_vec );
                
                // print output
                for (int T = 0; T < res_out.N.size(); T += p.gen_step) {
                    std::cout << i+1 << ";" << T;
                    
                    // iterate over classes
                    for (int cls = 0; cls < p.n_classes; cls++) {
                        std::cout << ";" << res_out.at(T, cls);
                    }
                    std::cout << ";" << res_out.N[T] << "\n";
                }
                
                // print out last line of each replicate
                int T_end = res_out.N.size() - 1;
                endResults << i+1 << ";" << T_end;
                for (int cls = 0; cls < p.n_classes; cls++) {
                    endResults << ";" << res_out.at(T_end, cls);
                }
                endResults << ";" << res_out.N[T_end] << "\n";
                
                // evaluate extinction results
                ext += res_out.extinct; T_ext += res_out.T_extinct; crit += res_out.crit;
            }
            break;
            
        // write to file
        case 1:
            
            // column headers
            simResults << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                simResults << ";" << class_names[cls];
            }
            simResults << ";N" << std::endl;
            
            endResults << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                endResults << ";" << class_names[cls];
            }
            endResults << ";N" << std::endl;
            
            
            // model results by replicate
            for(int i = 0; i < n_reps; i++){
                // run model
                if(p.verbose > 0){ errOut << "Rep. " << i+1 << "\n"; }
                treeData res_out = branchingSim_byGen( errOut, p, prob_matrix, birth_vec );
                
                // print output
                for (int T = 0; T < res_out.N.size(); T += p.gen_step) {
                    simResults << i+1 << ";" << T;
                    
                    // iterate over classes
                    for (int cls = 0; cls < p.n_classes; cls++) {
                        simResults << ";" << res_out.at(T, cls);
                    }
                    simResults << ";" << res_out.N[T] << "\n";
                }
                
                // print out last line of each replicate
                int T_end = res_out.N.size() - 1;
                endResults << i+1 << ";" << T_end;
                for (int cls = 0; cls < p.n_classes; cls++) {
                    endResults << ";" << res_out.at(T_end, cls);
                }
                endResults << ";" << res_out.N[T_end] << "\n";
                
                // evaluate extinction results
                ext += res_out.extinct; T_ext += res_out.T_extinct; crit += res_out.crit;
            }
            break;
        
        // write to file to ensure output will be obtainable
        default:
            
            simResults << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                simResults << ";" << class_names[cls];
            }
            simResults << ";N" << std::endl;
            
            endResults << "rep;" << "Gen";
            for (int cls = 0; cls < p.n_classes; cls++) {
                endResults << ";" << class_names[cls];
            }
            endResults << ";N" << std::endl;
            
            
            // model results by replicate
            for(int i = 0; i < n_reps; i++){
                // run model
                if(p.verbose > 0){ errOut << "Rep. " << i+1 << "\n"; }
                treeData res_out = branchingSim_byGen( errOut, p, prob_matrix, birth_vec );
                
                // print output
                for (int T = 0; T < res_out.N.size(); T += p.gen_step) {
                    simResults << i+1 << ";" << T;
                    
                    // iterate over classes
                    for (int cls = 0; cls < p.n_classes; cls++) {
                        simResults << ";" << res_out.at(T, cls);
                    }
                    simResults << ";" << res_out.N[T] << "\n";
                    
                }
                
                // print out last line of each replicate
                int T_end = res_out.N.size() - 1;
                endResults << i+1 << ";" << T_end;
                for (int cls = 0; cls < p.n_classes; cls++) {
                    endResults << ";" << res_out.at(T_end, cls);
                }
                endResults << ";" << res_out.N[T_end] << "\n";
                
                // evaluate extinction results
                ext += res_out.extinct; T_ext += res_out.T_extinct; crit += res_out.crit;
                
            }
            break;
    }
    T_ext = double(T_ext / ext); // mean extinction time given extinction
    ext = double(ext / n_reps); // fraction of simulations where extinction occured
    crit = double(crit / n_reps); // fraction of simulations crossing critical threshold
    extinct << ext << "; " << T_ext << ";" << crit << std::endl;

    extinct.close(); // extinction metrics
    simResults.close(); // result file
    endResults.close(); // endpoint file
    errOut.close(); // error file

    return 0;
    
}


