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
#include <cassert>

#include "gsl/gsl_randist.h"

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

// struct to hold parameter value inputs

//Struct for storing the various parameters that go into the model.
struct Parameters
{
    int n_WT = 1; // initial infections
    int n_M1 = 0;
    int n_M2 = 0;
    int n_M12 = 0;
    
    double b = 1.0; // birth rate -- assume all equal
    double d = 1.0; // death rate -- assume all equal
    double mu = 0.0; // mutation probability
    
    double T_max = 10.0; // number of generations / events / or time interval depending on model structure
    int data_out = 0; // specify model type. 0: exact continuous time; 1: stepwise; 2: synchronous discrete
    std::string batchname = "outSize_distn"; // output file name tag
    
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

        size_t conf_length = 10; //number of model input parameters
        if (conf_v.size()!=conf_length)
        {
            std::cout << "Expected configuration file with " << conf_length << " options, loaded file with "
                      << conf_v.size() << " lines." << std::endl;
            exit(EXIT_FAILURE);
        }

        n_WT = std::stoi(conf_v[0]); // initial infections
        n_M1 = std::stoi(conf_v[1]);
        n_M2 = std::stoi(conf_v[2]);
        n_M12 = std::stoi(conf_v[3]);
        
        b = std::stod(conf_v[4]); // birth rate -- assume all equal
        d = std::stod(conf_v[5]); // death rate -- assume all equal
        mu = std::stod(conf_v[6]); // mutation probability
        
        T_max = std::stoi(conf_v[7]);
        data_out = std::stoi(conf_v[8]); // output indicator
        batchname = conf_v[9]; // output file name tag
       
    }
};

//
struct treeData{
    
    std::vector<int> n_WT; // number of active wild type nodes
    std::vector<int> n_M1; // number of active M1 nodes
    std::vector<int> n_M2; // number of active M2 nodes
    std::vector<int> n_M12; // number of active M12 nodes
    std::vector<double> time; // current time vector for use in continuous time process
    
    treeData(int n_steps){
        
        n_WT.assign(n_steps, 0); // number of active wild type nodes
        n_M1.assign(n_steps, 0); // number of active M1 nodes
        n_M2.assign(n_steps, 0); // number of active M2 nodes
        n_M12.assign(n_steps, 0); // number of active M12 nodes
        time.assign(n_steps, 0); // current simulation time
    }
    
};


// Function runs a full gillespie algorithm
// This models the exact continuous time process to be
// compared against the generating function ODE system
treeData branchingSim_CTMC( std::ofstream &errOut, const Parameters& p ){
    
    treeData data(1); // vectors of size 1 to hold initial state, but dynamically resized to account for unknown number of events

    // simulation values
    int WT = p.n_WT;
    int M1 = p.n_M1;
    int M2 = p.n_M2;
    int M12 = p.n_M12;
    int N = WT + M1 + M2 + M12;
    double T = 0;
    
    
    data.n_WT[0] = WT; // number of initial wild type nodes
    data.n_M1[0] = M1; // number of initial wild type nodes
    data.n_M2[0] = M2; // number of initial wild type nodes
    data.n_M12[0] = M12; // number of initial wild type nodes
    data.time[0] = 0; // initial time
    
    while(T <= p.T_max && N > 0){
        
        double P_birth_tot = double(N) * p.b;
        double P_death_tot = double(N) * p.d;
        double rate_frac = 1.0 / (P_birth_tot + P_death_tot);
        double N_frac = 1.0 / double(N);
        double P_mu_0 = pow(1.0 - p.mu, 2.0); // probablity of no mutations
        double P_mu_1 = p.mu*(1.0 - p.mu); // probability of exactly 1 mutation
        double U1, U2, U3; // uniform RVs to determine event type
        
        T += draw_exp_gsl(rate_frac);
        
        // determine next event as birth or death
        U1 = draw_uniform_gsl(0.0, 1.0);
        if(U1 < (P_birth_tot) * rate_frac){ // birth event
            U2 = draw_uniform_gsl(0.0, 1.0); // determining parent type
            U3 = draw_uniform_gsl(0.0, 1.0); // determining offspring type given parent type
            
            if(U2 < WT * N_frac){ // WT parent
                
                if(U3 < P_mu_0){ // WT offspring
                    WT++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // M2 offspring
                    M2++;
                }else{ // M12 offspring
                    M12++;
                }
                
            }else if(U2 < (WT + M1) * N_frac){ // M1 parent
                
                if(U3 < P_mu_0){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // WT offspring
                    WT++;
                }else{ // M2 offspring
                    M2++;
                }
                
            }else if(U2 < (WT + M1 + M2) * N_frac){ // M2 parent
                
                if(U3 < P_mu_0){ // M2 offspring
                    M2++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // WT offspring
                    WT++;
                }else{ // M1 offspring
                    M1++;
                }
                
            }else{ // M12 parent
                
                if(U3 < P_mu_0){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // M2 offspring
                    M2++;
                }else{ // WT offspring
                    WT++;
                }
                
            }
            
        }else{ // death event
            U2 = draw_uniform_gsl(0.0, 1.0);
            if(U2 < WT * N_frac){ // WT death
                WT--;
            }else if(U2 < (WT + M1) * N_frac){ // M1 death
                M1--;
            }else if(U2 < (WT + M1 + M2) * N_frac){ // M2 death
                M2--;
            }else{ // M12 death
                M12--;
            }
            
        }
        
        N = WT + M1 + M2 + M12; // update N
        
        data.n_WT.push_back(WT); // number of current wild type nodes
        data.n_M1.push_back(M1); // number of current wild type nodes
        data.n_M2.push_back(M2); // number of current wild type nodes
        data.n_M12.push_back(M12); // number of current wild type nodes
        data.time.push_back(T);
        
    } // end while loop
    
    return(data);
    
}


// Function runs a full gillespie algorithm
// This models the embedded jump chain in the continuous time process
// but does not correspond with the continuous time ODE system
treeData branchingSim_discreteStepwise( std::ofstream &errOut, const Parameters& p ){
    
    // T_max must be cast as int since it is encoded as a double for use in continuous time simulations
    treeData data(int(p.T_max)+1);

    // simulation values
    int WT = p.n_WT;
    int M1 = p.n_M1;
    int M2 = p.n_M2;
    int M12 = p.n_M12;
    int N = WT + M1 + M2 + M12;
    int T = 0;
    
    
    data.n_WT[T] = WT; // number of initial wild type nodes
    data.n_M1[T] = M1; // number of initial wild type nodes
    data.n_M2[T] = M2; // number of initial wild type nodes
    data.n_M12[T] = M12; // number of initial wild type nodes
    T++;
    
    while(T <= int(p.T_max) && N > 0){
        
        double P_birth_tot = double(N) * p.b;
        double P_death_tot = double(N) * p.d;
        double rate_frac = 1.0 / (P_birth_tot + P_death_tot);
        double N_frac = 1.0 / double(N);
        double P_mu_0 = pow(1.0 - p.mu, 2.0); // probablity of no mutations
        double P_mu_1 = p.mu*(1.0 - p.mu); // probability of exactly 1 mutation
        double U1, U2, U3; // uniform RVs to determine event type
        
        // determine next event as birth or death
        U1 = draw_uniform_gsl(0.0, 1.0);
        if(U1 < (P_birth_tot) * rate_frac){ // birth event
            U2 = draw_uniform_gsl(0.0, 1.0); // determining parent type
            U3 = draw_uniform_gsl(0.0, 1.0); // determining offspring type given parent type
            
            if(U2 < WT * N_frac){ // WT parent
                
                if(U3 < P_mu_0){ // WT offspring
                    WT++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // M2 offspring
                    M2++;
                }else{ // M12 offspring
                    M12++;
                }
                
            }else if(U2 < (WT + M1) * N_frac){ // M1 parent
                
                if(U3 < P_mu_0){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // WT offspring
                    WT++;
                }else{ // M2 offspring
                    M2++;
                }
                
            }else if(U2 < (WT + M1 + M2) * N_frac){ // M2 parent
                
                if(U3 < P_mu_0){ // M2 offspring
                    M2++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // WT offspring
                    WT++;
                }else{ // M1 offspring
                    M1++;
                }
                
            }else{ // M12 parent
                
                if(U3 < P_mu_0){ // M12 offspring
                    M12++;
                }else if(U3 < P_mu_0 + P_mu_1){ // M1 offspring
                    M1++;
                }else if(U3 < P_mu_0 + 2.0*P_mu_1){ // M2 offspring
                    M2++;
                }else{ // WT offspring
                    WT++;
                }
                
            }
            
        }else{ // death event
            U2 = draw_uniform_gsl(0.0, 1.0);
            if(U2 < WT * N_frac){ // WT death
                WT--;
            }else if(U2 < (WT + M1) * N_frac){ // M1 death
                M1--;
            }else if(U2 < (WT + M1 + M2) * N_frac){ // M2 death
                M2--;
            }else{ // M12 death
                M12--;
            }
            
        }
        
        N = WT + M1 + M2 + M12; // update N
        
        data.n_WT[T] = WT; // number of current wild type nodes
        data.n_M1[T] = M1; // number of current wild type nodes
        data.n_M2[T] = M2; // number of current wild type nodes
        data.n_M12[T] = M12; // number of current wild type nodes
        T++;
        
    } // end while loop
    
    return(data);
    
}



// simulation code for discrete branching process
// this corresponds to the discrete time generating function
treeData branchingSim_byGen( std::ofstream &errOut, const Parameters& p ){
    
    // T_max must be cast as int since it is encoded as a double for use in continuous time simulations
    treeData data(int(p.T_max)+1);

    // initialize counts
    int WT  = p.n_WT;
    int M1  = p.n_M1;
    int M2  = p.n_M2;
    int M12 = p.n_M12;

    data.n_WT[0]  = WT;
    data.n_M1[0]  = M1;
    data.n_M2[0]  = M2;
    data.n_M12[0] = M12;
    
    // event rates -- offspring probabilities
    double p_birth = p.b / (p.b + p.d);
    double p_mu_0 = pow(1.0 - p.mu, 2.0);
    double p_mu_1 = p.mu*(1.0 - p.mu);
    double p_mu_2 = p_mu_1;
    double p_mu_3 = pow(p.mu,2.0);
    std::vector<double> probs = {p_mu_0, p_mu_1, p_mu_2, p_mu_3};
    
    // offspring mapping array
    // rows represent parent type
    // column values(j) represent the mapping of
    // p_mu_j for the corresponding offspring type
    std::array<std::array<int,4>,4> mapping = {{
            {0,1,2,3}, // WT
            {1,0,3,2}, // M1
            {2,3,0,1}, // M2
            {3,2,1,0}  // M12
        }};
    
    // offspring probability array
    std::array<std::array<double,4>,4> mu_mat; // 4x4 array
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mu_mat[i][j] = probs[mapping[i][j]];
        }
    }
    
    // iterate over T_max generations
    for (int T = 0; T < int(p.T_max); ++T) {
        std::array<int,4> current_gen = {WT, M1, M2, M12};
        std::array<int,4> next_gen = {0,0,0,0};

        // Loop over types
        for (int type = 0; type < 4; ++type) {
            if (current_gen[type] == 0) continue;

            // number of individuals that reproduce this generation
            int n_reproduce = draw_binom_gsl(current_gen[type], p_birth);
            if (n_reproduce == 0) continue;
            
            // determine offspring distribution of n_reproduce parents of type 'type'
            std::vector<int> offspring_vector = draw_multinom_gsl(n_reproduce, std::vector<double>(mu_mat[type].begin(), mu_mat[type].end()) );

            // Add offspring to next generation
            for (int j = 0; j < offspring_vector.size(); ++j) {
                next_gen[j] += offspring_vector[j];
            }

            // Add the parents back to their own type
            next_gen[type] += n_reproduce;
            
            /*
            for (int i = 0; i < n_reproduce; ++i) {
                int pattern = draw_discrete_gsl(probs); // which offspring type pattern
                if (pattern == 0) { // no mutations
                    next_gen[mapping[type][0]] += 2; // two parent-type offspring
                } else { // offspring type dictated by "pattern" RV
                    next_gen[mapping[type][0]] += 1; // surviving parent
                    next_gen[mapping[type][pattern]] += 1; // resulting offspring
                }
            }
            */
        }

        // update counts for next generation
        WT  = next_gen[0];
        M1  = next_gen[1];
        M2  = next_gen[2];
        M12 = next_gen[3];

        // store current counts
        data.n_WT[T+1]  = WT;
        data.n_M1[T+1]  = M1;
        data.n_M2[T+1]  = M2;
        data.n_M12[T+1] = M12;

        // early stop if extinct
        if (WT + M1 + M2 + M12 == 0) break;
    }
    
    return(data);
    
}



int main(int argc, char* argv[]){
    
    // initialize parameters read from command line input
    int n_reps = std::stoi(argv[1]); // simulation replicates
    Parameters p(argc, argv);
    
    // open error file
    std::string err_fname = "./src/simulation_files/" + p.batchname + "_err.txt";
    std::ofstream errOut(err_fname);
    //assert(errOut.is_open());
    
    // open full output file
    std::string fname = "./src/simulation_files/" + p.batchname + "_fullSim.txt";
    std::ofstream simResults(fname);
    //assert(simResults.is_open());
    
    
    switch (p.data_out) {
        case 0:
            // write headers for continuous time model
            std::cout << "rep;" << "idx" << "time;" << "n_WT;" << "n_M1;" << "n_M2;" << "n_M12" << std::endl;
            
            // model results
            for(int i = 0; i < n_reps; i++){
                treeData res_out = branchingSim_CTMC( errOut, p );
                for(int idx = 0; idx < res_out.n_WT.size(); idx++){
                    std::cout << i+1 << ";" << idx << ";" << res_out.time[idx] << ";" << res_out.n_WT[idx] << ";" << res_out.n_M1[idx] << ";" << res_out.n_M2[idx] << ";" << res_out.n_M12[idx] <<  std::endl;
                }
            }
            break;
            
        case 1:
            
            // write headers for discrete stepwise process
            std::cout << "rep;" << "step;" << "n_WT;" << "n_M1;" << "n_M2;" << "n_M12" << std::endl;
            
            // model results
            for(int i = 0; i < n_reps; i++){
                treeData res_out = branchingSim_discreteStepwise( errOut, p );
                for(int T = 0; T < res_out.n_WT.size(); T++){
                    std::cout << i+1 << ";" << T << ";" << res_out.n_WT[T] << ";" << res_out.n_M1[T] << ";" << res_out.n_M2[T] << ";" << res_out.n_M12[T] <<  std::endl;
                }
            }
            break;
            
        case 2:
            // write headers
            std::cout << "rep;" << "Gen;" << "n_WT;" << "n_M1;" << "n_M2;" << "n_M12" << std::endl;
            
            // model results
            for(int i = 0; i < n_reps; i++){
                treeData res_out = branchingSim_byGen( errOut, p );
                for(int T = 0; T < res_out.n_WT.size(); T++){
                    std::cout << i+1 << ";" << T << ";" << res_out.n_WT[T] << ";" << res_out.n_M1[T] << ";" << res_out.n_M2[T] << ";" << res_out.n_M12[T] <<  std::endl;
                }
            }
            break;
            
        default:
            std::cout << "Error: invalid value for data_out: " << p.data_out << ". Valid options are {0, 1, 2}." << std::endl;
            break;
    }
    
    
    simResults.close(); // result file
    errOut.close(); // error file

    return 0;
    
}


