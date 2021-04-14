/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  num_particles = 101;  // TODO: Set the number of particles
  //resize weights to the size of number of particles 
  weights.resize(num_particles, 1.0);
  Particle particle; 
  for(int i=0; i<num_particles;++i){
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen); 
    particle.theta = dist_theta(gen); 
    particle.weight = 1.0; 
    particles.push_back(particle);
  }
  
  // Now the particle filter is initialized
  is_initialized  = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for(auto &p : particles){
    if (std::fabs(yaw_rate) < 0.0001){
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }
    else{
      p.x = p.x + velocity*(sin(p.theta + yaw_rate*delta_t)-sin(p.theta))/yaw_rate; 
      p.y = p.y + velocity*(cos(p.theta)- cos(p.theta + yaw_rate*delta_t))/yaw_rate; 
      p.theta = p.theta + yaw_rate*delta_t; 
    }

    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
    p.x += dist_x(gen); 
    p.y += dist_y(gen);
    p.theta += dist_theta(gen); 
  } 
  
  //particles.x = particles.x + velocity*(sin(theta + yaw_rate*delta_t)-sin(theta))/yaw_rate; 
  //p.y = p.y + velocity*(cos(theta)- cos(theta + yaw_rate*delta_t))/yaw_rate; 
  //p.theta = p.theta + yaw_rate*delta_t; 
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto obs : observations){
    double lowest_dist = std::numeric_limits<double>::max();
    double closest_id = -1; 
    double obs_x = obs.x;
    double obs_y = obs.y;
    for(auto pred : predicted){
      double pred_x = pred.x;
      double pred_y = pred.y;
      int pre_id = pred.id;
      double distance = dist(pred_x, pred_y, obs_x,obs_y);  
      if(distance < lowest_dist){
        lowest_dist = distance; 
        closest_id = pre_id; 
      }
      //assign the nearest measurement id to the observed landmarks
      obs.id = closest_id; 
    }
  }  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
 
  for(auto &p : particles){
    double particle_x = p.x;
    double particle_y = p.y;
    double particle_theta = p.theta; 
    //reset the particle weights to 1 
    p.weight = 1.0; 
	//transform every observation into map coordinates
    std::vector<LandmarkObs> transformed_observations;
    for(auto obs: observations){
      LandmarkObs transformed_obs;
      transformed_obs.x = particle_x + cos(particle_theta)*obs.x - sin(particle_theta)*obs.y;
      transformed_obs.y = particle_y + cos(particle_theta)*obs.y + sin(particle_theta)*obs.x; 
      transformed_obs.id = obs.id; 
      transformed_observations.push_back(transformed_obs);
    }
    
    std::vector<LandmarkObs> predicted; 
    //keep only the landmarks that are in a sensor range. 
    for(auto land : map_landmarks.landmark_list){
      double land_distance = dist(p.x,p.y,land.x_f,land.y_f);
      if(land_distance < sensor_range){
        LandmarkObs predicted_landmark; 
        predicted_landmark.id = land.id_i; 
        predicted_landmark.x = land.x_f;
        predicted_landmark.y = land.y_f; 
        predicted.push_back(predicted_landmark); 
      }
    }
    //find the closest predicted meausrement 
    dataAssociation(predicted,transformed_observations); 
    //Calculate Multivariate function 
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double var_x = sigma_x*sigma_x;
    double var_y = sigma_y*sigma_y; 
    double gaus_norm = 1.0/(2.0*sigma_x*sigma_y*M_PI); 
    double weight = 1.0;
    double mx, my; 
    double probability = 1.0; 
    double weights_sum = 0.0; 
    for(int j = 0; j < transformed_observations.size();++j){
      double tf_x = transformed_observations[j].x; 
      double tf_y = transformed_observations[j].y; 
      double tf_id = transformed_observations[j].id; 
      for(int k = 0; k < predicted.size(); ++k){
        double pd_x = predicted[k].x; 
        double pd_y = predicted[k].y;
        double pd_id = predicted[k].id; 
        if(tf_id == pd_id){
          mx = pd_x; 
          my = pd_y;
          //double exponent = (dx*dx)/(2.0*var_x)+ (dy*dy)/(2.0*sigma_y);
          //weight = gaus_norm*exp(-exponent);
          //probability *= weight; 
          break; 
        }
      }
      double dx = tf_x - mx; 
      double dy = tf_y - my;
      double exponent = (dx*dx)/(2.0*var_x)+ (dy*dy)/(2.0*sigma_y);
      weight = gaus_norm*exp(-exponent);
      probability *= weight; 
    }
    p.weight = probability; 
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampledParticles;
  vector<double> weights; 
  
  for (size_t i = 0; i < num_particles; i++) {
     weights.push_back(particles[i].weight);
   }

  //generate random weights 
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  for(int j = 0; j < particles.size();++j){
    int index = dist(gen); 
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles; 

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}