

		###############################################################
		##															 ##
		##			Multi-Core Parallel Tempering 				 	 ##
		##		   (with markov chain monte carlo)					 ##
		##                                                           ##
		###############################################################


# Multicore Parallel Tempering with Random Walk MCMC for Weighted Mixture of Distributions for  Curve Fitting.
# Ratneel Deo and Rohitash Chandra (2018).
# SCIMS, USP. deo.ratnee@gmail.com, CTDS, UniSYD. c.rohitash@gmail.com
# Simulated data is used.


from __future__ import print_function, division
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.mlab as mlab
import time
import os



#-----------------------------------------------------------------------
# Define forward model (misure of models finction used)
#-----------------------------------------------------------------------
def fx_func(nModels, x, mu, sig, w):
	fx = np.zeros(x.size)
	for i in range(nModels):
		fx = fx + w[i] * mlab.normpdf(x, mu[i], np.sqrt(sig[i]))
	return fx


#-----------------------------------------------------------------------
# Define the MCMC class  
# 	- MCMC is defined as process to enable multicore implemantation
#-----------------------------------------------------------------------

class BayesMCMC(multiprocessing.Process):  # MCMC class

	def __init__(self, samples, nModels, ydata, tempr,  x, parameter_queue, event, main_proc,step_size_mu,step_size_nu,step_size_eta,swap_interval):


		#-----------------------------------------------------------------------
		# initialize the process and  define queues and events
		#-----------------------------------------------------------------------
		multiprocessing.Process.__init__(self)
		self.processID = tempr		
		self.parameter_queue = parameter_queue
		self.event = event
		self.signal_main = main_proc
		self.temprature = tempr



		#-----------------------------------------------------------------------
		# initialize parameters
		#-----------------------------------------------------------------------
		self.samples = samples
		self.nModels = nModels 
		self.x = x 
		self.ydata = ydata 
		self.start_chain = 0, 
		self.end = 0
		self.step_size_mu = step_size_mu  # need to choose these values according to the problem
		self.step_size_nu = step_size_nu
		self.step_size_eta = step_size_eta
		self.swap_interval = swap_interval


		

		#-----------------------------------------------------------------------
		# create posterior arrays for MCMC results
		#-----------------------------------------------------------------------
		self.fx_samples = np.ones((samples, ydata.size))
		self.pos_mu = np.ones((samples, nModels))
		self.pos_sig = np.ones((samples, (nModels)))
		self.pos_w = np.ones((samples, (nModels)))
		self.pos_tau = np.ones((samples,  1))


		#-----------------------------------------------------------------------
		# create initial MCMC parameters and calculate  Priors
		#-----------------------------------------------------------------------
		self.mu_current = np.zeros(nModels)
		self.sig_current = np.zeros(nModels)  # to get sigma
		self.nu_current = np.zeros(nModels)  # to get sigma
		self.w_current = np.zeros(nModels)		
		for i in range(nModels):
			self.sig_current[i] = np.var(x)
			self.mu_current[i] = np.mean(x)
			self.nu_current[i] = np.log(self.sig_current[i])
			self.w_current[i] = 1.0 / nModels
		fx = fx_func(nModels, x, self.mu_current, self.sig_current, self.w_current)
		t = np.var(fx - ydata)
		self.tau_current = t
		self.eta_current = np.log(t) 
		self.lhood = 0
		self.naccept = 0		
		self.likelihood_current, fx = self.likelihood_func_pt(nModels, x, ydata, self.mu_current,
												 self.sig_current, self.w_current,
												 self.tau_current)
		

	#-----------------------------------------------------------------------
	# liklihood function incoorporates temprature for parallel tempering
	#-----------------------------------------------------------------------
	def likelihood_func_pt(self, nModels, x, y, mu, sig, w, tau):
		tausq = tau
		fx = fx_func(nModels, x, mu, sig, w)
		loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
		#print(np.sum(loss))
		likelihood = (np.sum(loss))* (1.0/self.temprature)
		return [likelihood, fx]
	

	#-----------------------------------------------------------------------
	# propose parameters for one sample of MCMC chain
	#-----------------------------------------------------------------------	
	def propose_parameters(self, nModels):
		weights = []
		if nModels == 1:
				weights = [1]
		elif nModels == 2:
			# (genreate vector that  adds to 1)
			weights = np.random.dirichlet((1, 1), 1)
		elif nModels == 3:
			weights = np.random.dirichlet((1, 1, 1), 1)  # (vector adds to 1)
		elif nModels == 4:
			weights = np.random.dirichlet(
				(1, 1, 1, 1), 1)  # (vector adds to 1)
		elif nModels == 5:
			weights = np.random.dirichlet(
				(1, 1, 1, 1, 1), 1)  # (vector adds to 1)
				
		return weights       
				
	
	#-----------------------------------------------------------------------
	# run the MCMC chain with initial priors
	#-----------------------------------------------------------------------
	def run (self):
		mu_proposal = np.zeros(self.nModels)
		sig_proposal = np.zeros(self.nModels)  # sigma
		nu_proposal = np.zeros(self.nModels)
		w_proposal = np.zeros(self.nModels)
		eta_proposal = 0.1
		tau_proposal = 0   
		
		
		# run for the number of sample in the MCMC chain 
		for i in range(self.start_chain, self.end, 1):
   
			##create a new set of parapamers (theta)
			weights = self.propose_parameters(self.nModels)

			nu_proposal = self.nu_current + np.random.normal(0, self.step_size_nu, self.nModels)
			sig_proposal = np.exp(nu_proposal)
			mu_proposal = self.mu_current + np.random.normal(0, self.step_size_mu, self.nModels)


			#propose a new set of weights (randomly generate the values of theta)
			for j in range(self.nModels):
				
				# ensure they stay between a range
				if mu_proposal[j] < 0 or mu_proposal[j] > 1:
					mu_proposal[j] = random.uniform(np.min(self.x), np.max(self.x))

				w_proposal[j] = weights[0, j]  # just for vector consistency

			eta_proposal = self.eta_current + np.random.normal(0, self.step_size_eta, 1)
			tau_proposal = math.exp(eta_proposal)

			likelihood_proposal, fx = self.likelihood_func_pt(self.nModels, self.x, self.ydata,
													  mu_proposal, sig_proposal,
													  w_proposal, tau_proposal)

			diff = likelihood_proposal - self.likelihood_current

			mh_prob = min(1, math.exp(diff))

			u = random.uniform(0, 1)

			

			if u < mh_prob:
				# Update position
				#print(i, ' is accepted sample')
				self.naccept += 1
				self.likelihood_current = likelihood_proposal
				self.mu_current = mu_proposal
				self.nu_current = nu_proposal
				self.eta_current = eta_proposal

				#print(self.likelihood_current, self.mu_current, self.nu_current,  self.eta_current, 'accepted')
				#print(self.mu_proposal)
				
				self.pos_mu[i + 1, ] = mu_proposal
				self.pos_sig[i + 1, ] = sig_proposal
				self.pos_w[i + 1, ] = w_proposal
				self.pos_tau[i + 1, ] = tau_proposal
				self.fx_samples[i + 1, ] = fx
				#print (self.pos_mu)

			else:
				#print('here', self.pos_mu[i+1,])
				
				self.pos_mu[i + 1, ] = self.pos_mu[i, ]
				self.pos_sig[i + 1, ] = self.pos_sig[i, ]
				self.pos_w[i + 1, ] = self.pos_w[i, ]
				self.pos_tau[i + 1, ] = self.pos_tau[i, ]
				self.fx_samples[i + 1, ] = self.fx_samples[i, ]
				
			

			if ( i % self.swap_interval == 0 ): 

				self.lhood = self.likelihood_current		
				param = [self.mu_current, self.nu_current, self.eta_current, self.lhood]        

				# paramater placed in queue for swapping between chains
				self.parameter_queue.put(param)
				
			    
			    #signal main process to start and start waiting for signal for main
				self.signal_main.set()				
				self.event.wait()
				

				# retrieve parametsrs fom ques if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result =  self.parameter_queue.get()
						
						self.mu_current = result[0]
						self.nu_current = result[1]
						self.eta_current = result[2]
						self.lhood = result[3]
					except:
						print ('error')

		# end for
		#------------------------------------------------------------------------------

		self.lhood = self.likelihood_current
		print(self.pid, self.naccept / self.samples, '% was accepted')

		
		param = [self.mu_current, self.nu_current, self.eta_current, self.lhood]        
		self.parameter_queue.put(param)



		# save posterior for all the chains
		file_name = 'posterior/pos_mu/chain_'+ str(self.temprature)+ '.txt'
		np.savetxt(file_name,self.pos_mu)
		file_name = 'posterior/pos_sig/chain_'+ str(self.temprature)+ '.txt'
		np.savetxt(file_name,self.pos_sig)
		file_name = 'posterior/pos_w/chain_'+ str(self.temprature)+ '.txt'
		np.savetxt(file_name,self.pos_w)
		file_name = 'posterior/pos_tau/chain_'+ str(self.temprature)+ '.txt'
		np.savetxt(file_name,self.pos_tau)
		
		###############################################################
		##															 ##
		##			save this file only if you need it 				 ##
		##				(may get really large)						 ##
		##                                                           ##
		###############################################################
		file_name = 'posterior/fx_samples/chain_'+ str(self.temprature)+ '.txt'
		#np.savetxt(file_name,self.fx_samples)

		# signal main process to resume
		self.signal_main.set()
		

class ParallelTempering:

	def __init__(self, num_chains, maxtemp,NumSample,ydata,nModels,step_size_mu,step_size_nu,step_size_eta,swap_interval):

		self.step_size_mu = step_size_mu  # need to choose these values according to the problem
		self.step_size_nu = step_size_nu
		self.step_size_eta = step_size_eta
		self.swap_interval = swap_interval
		
		self.maxtemp = maxtemp
		self.num_chains = num_chains
		self.chains = []
		self.tempratures = []
		self.NumSamples = int(NumSample/self.num_chains)
		self.sub_sample_size = max(1, int( 0.1* self.NumSamples))
		
		# create queues for transfer of parameters between process chain
		self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]

		# two ways events are used to synchronize chains
		self.event = [multiprocessing.Event() for i in range (self.num_chains)]
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]


		  
	
	# assigin tempratures dynamically   
	def assign_temptarures(self):
		tmpr_rate = (self.maxtemp /self.num_chains)
		temp = 1
		for i in xrange(0, self.num_chains):            
			self.tempratures.append(temp)
			temp += tmpr_rate
			print(self.tempratures[i])
			
	
	# Create the chains.. Each chain gets its own temprature
	def initialize_chains (self, nModels, ydata,x):
		self.assign_temptarures()
		for i in xrange(0, self.num_chains):
			self.chains.append(BayesMCMC(self.NumSamples, nModels,ydata, self.tempratures[i],x ,self.chain_parameters[i], self.event[i], self.wait_chain[i],self.step_size_mu,self.step_size_nu,self.step_size_eta,self.swap_interval ))
			
			
	
		
	# Merge different MCMC chains y stacking them on top of each other       
	def merge_chain (self, chain):
		print (chain.shape)
		print (self.NumSamples, int(chain[0].size/chain[0][0].size))

		comb_chain = []
		for i in xrange(0, self.num_chains):
			for j in xrange(0, self.NumSamples):
				comb_chain.append(chain[i][j].tolist())     
		return np.asarray(comb_chain)
		

	def run_chains (self, nModels, x, ydata):
		
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		
		# create parameter holders for paramaters that will be swapped
		chain_mu_current = np.zeros((self.num_chains,nModels))
		chain_nu_current = np.zeros((self.num_chains,nModels))
		chain_eta_current = np.zeros((self.num_chains,1))
		lhood = np.zeros(self.num_chains)

		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.NumSamples-1


		
		#-------------------------------------------------------------------------------------
		# intialize the MCMC chains 
		#-------------------------------------------------------------------------------------
		self.initialize_chains ( nModels, ydata,x)


		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for l in range(0,self.num_chains):
			self.chains[l].start_chain = start
			self.chains[l].end = end
		
		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for j in range(0,self.num_chains):        
			self.chains[j].start()



		flag_running = True 

		
		while flag_running:          

			#-------------------------------------------------------------------------------------
			# wait for chains to complete one pass through the samples
			#-------------------------------------------------------------------------------------

			for j in range(0,self.num_chains): 
				#print (j, ' - waiting')
				self.wait_chain[j].wait()
			

			
			#-------------------------------------------------------------------------------------
			#get info from chains
			#-------------------------------------------------------------------------------------
			
			for j in range(0,self.num_chains): 
				if self.chain_parameters[j].empty() is False :
					result =  self.chain_parameters[j].get()
					#print(result)
					chain_mu_current[j] = result[0]
					chain_nu_current[j] = result[1]
					chain_eta_current[j] = result[2]
					lhood[j] = result[3]
				

			#-------------------------------------------------------------------------------------
			# propose swapping using likelihoods
			#-------------------------------------------------------------------------------------

			# create swapping proposals between adjacent chains
			for k in range(0, self.num_chains-1): 
				swap_proposal[k]=  (lhood[k]/[1 if lhood[k+1] == 0 else lhood[k+1]])*(1/self.tempratures[k] * 1/self.tempratures[k+1])  



			for l in range( self.num_chains-1, 0, -1):            
				u = 100000#random.uniform(0, 1) 
				swap_prob = min(1, swap_proposal[l-1])

				#randomly choose to accep to reject swap based on swap proposal
				if u < swap_prob : 
					
					#swap parameters between adjacent chains 
					para = [chain_mu_current[l-1], chain_nu_current[l-1], chain_eta_current[l-1] ,lhood[j-1]  ]
					self.chain_parameters[l].put(para)
					param = [chain_mu_current[l], chain_nu_current[l], chain_eta_current[l],lhood[j] ]
					self.chain_parameters[l-1].put(param)
					
				else:
					para = [chain_mu_current[l-1], chain_nu_current[l-1], chain_eta_current[l-1] ,lhood[j-1]  ]
					self.chain_parameters[l-1].put(para)
					param = [chain_mu_current[l], chain_nu_current[l], chain_eta_current[l],lhood[j] ]
					self.chain_parameters[l].put(param)


			#-------------------------------------------------------------------------------------
			# resume suspended process
			#-------------------------------------------------------------------------------------
			for k in range (self.num_chains):
					self.event[k].set()
								

			#-------------------------------------------------------------------------------------
			#check if all chains have completed runing
			#-------------------------------------------------------------------------------------
			count = 0
			for i in range(self.num_chains):
				if self.chains[i].is_alive() is False:
					count+=1
					while self.chain_parameters[i].empty() is False:
						dummy = self.chain_parameters[i].get()

			if count == self.num_chains :
				flag_running = False
			

		#-------------------------------------------------------------------------------------
		#wait for all processes to jin the main process
		#-------------------------------------------------------------------------------------	   
		for j in range(0,self.num_chains): 
			self.chains[j].join()

		print('process ended') 
		

		#-------------------------------------------------------------------------------------
		# recover posterior chains and merge into a single posterior chain
		#-------------------------------------------------------------------------------------			
		data  = []

		for i in range (self.num_chains):
			file_name = 'posterior/pos_mu/chain_'+ str(self.tempratures[i])+ '.txt'
			file = np.loadtxt(file_name)
			data.append(file)

		chain_data = self.merge_chain(np.asarray(data))

		
		
		return chain_data 
	  


#plot a figure of posterior distributions in histogram
def plot_figure(list_points,title,ylabel,xlabel):
	bins = np.linspace(0, 1, 20)
	plt.clf()
	plt.hist(list_points, bins)
	plt.savefig( title + '.png')
	plt.show()

	
def make_directory (directory):	
	if not os.path.exists(directory):
		os.makedirs(directory)




def main():
	random.seed(time.time())
	nModels = 2

	
	make_directory('posterior/pos_mu')
	make_directory('posterior/pos_sig')
	make_directory('posterior/pos_w')
	make_directory('posterior/pos_tau')
	make_directory('posterior/fx_samples')

	# load univariate data in same format as given
	modeldata = np.loadtxt('simdata.txt')
	ydata = modeldata
  
	x = np.linspace(1 / ydata.size, 1, num=ydata.size)  # (input x for ydata)


	#-------------------------------------------------------------------------------------
	# Paramaters need to be finetuned for the problem being solved and the foward model being used
	#-------------------------------------------------------------------------------------

	NumSamples = 50000 # need to pick yourself
	
	# Number of chains of MCMC required to be run 
	# PT is a multicore implementation must num_chains >= 2
	# Choose a value less than the numbe of core available (avoid context swtiching)
	num_chains = 10

	#parameters for Parallel Tempering
	maxtemp = int(num_chains * 10)/2
	swap_interval = int(0.001 * NumSamples) #time when swap will be proposed ()


	# parameters for MCMC (need to be adjusted based on problem)
	step_size_mu = 0.1
	step_size_nu = 0.2
	step_size_eta = 0.1
	
	

	#-------------------------------------------------------------------------------------
	#Create A a Patratellel Tempring object instance 
	#-------------------------------------------------------------------------------------
	pt = ParallelTempering(num_chains, maxtemp, NumSamples,ydata,nModels,step_size_mu,step_size_nu,step_size_eta,swap_interval)

	#run the chains in a sequence in ascending order
	pos_mu = pt.run_chains( nModels, x, ydata)
	print('sucessfully sampled')
	

	#-------------------------------------------------------------------------------------
	#remove the initial burnin period of MCMC
	#-------------------------------------------------------------------------------------
	burnin = 0.2 * NumSamples   # use post burn in samples
	pos_mu = pos_mu[int(burnin):]
	
	#plot posterior distribution of MU 
	plot_figure(pos_mu[:,0],'Posterior_MU_Distrubution','y','x')
	

	
if __name__ == "__main__":
	main()
