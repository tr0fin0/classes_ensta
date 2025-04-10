import numpy as np
import matplotlib.pyplot as plt

class Bernoulli:
    """ Bernoulli Arm """

    def __init__(self,theta):
        # create a Bernoulli arm with mean theta
        self.mean = theta
        self.variance = theta*(1-theta)

    def sample(self):
        # generate a reward from a Bernoulli arm 
        return float(np.random.rand() < self.mean)


class Gaussian:
    """ Gaussian Arm """

    def __init__(self,mu,var=1):
        # create a Gaussian arm with specified mean and variance
        self.mean = mu
        self.variance = var

    def sample(self):
        # generate a reward from a Gaussian arm 
        return self.mean + sqrt(self.variance)*np.random.randn()
        

class Exponential:
    """ Exponential Arm """

    def __init__(self,p):
        # create an Exponential arm with parameter p
        self.mean = 1/p
        self.variance = 1/(p*p)

    def sample(self):
        # generate a reward from an Exponential arm 
        return -(self.mean)*np.log(np.random.rand())

class TruncatedExponential:
    """ Truncated Exponential Arm """

    def __init__(self,p,trunc):
        # create a truncated Exponential arm with parameter p
        self.p = p
        self.trunc = trunc
        self.mean = (1.-np.exp(-p * trunc)) / p
        self.variance=0
        
    def sample(self):
        # generate a reward from an Exponential arm 
        return min(-(1/self.p)*np.log(np.random.rand()),self.trunc)


class MixedMAB():
    """ Mixed-Arm Multi-Arm Bandit Machine """

    def __init__(self,arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.n_arms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def rwd(self,a):
        return self.arms[a].sample()


class GaussianMAB():
    """ Gaussian-Arm Multi-Arm Bandit Machine """

    def __init__(self,n_arms=2,labels=None,means=None):
        self.n_arms = n_arms
        if means is not None:
            self.means = means
        else:
            self.means = 0 + np.random.randn(self.n_arms)
        self.sigma = np.random.rand(self.n_arms) * 1
        if labels is None:
            labels = np.arange(n_arms)

    def rwd(self,a):
        return np.random.randn() * self.sigma[a] + self.means[a]
        
    def render(self, info=None): 
        pass


class BernoulliMAB():
    """ Bernoulli-Arm Multi-Arm Bandit Machine """

    def __init__(self,n_arms=2,labels=None,means=None):
        self.n_arms = n_arms
        if means is None:
            self.means = np.random.rand(self.n_arms)
        else:
            self.means = np.array(means)
        if labels is None:
            labels = np.arange(n_arms)
        self.labels = labels

    def rwd(self,a):
        return int(np.random.rand() < self.means[a]) 

    def render(self, info=None): 
        '''
        For generating figures
        '''
        fig = plt.figure(figsize=[5,4])
        ax = fig.add_subplot(111)

        plt.xticks(np.arange(self.n_arms), ['%d' % (_+1) for _ in range(self.n_arms)])
#        ax.set_yticks([0,1])
#        ax.set_yticklabels([0,1])
        ax.set_xticklabels(self.labels)
        ax.set_xlabel("action $A \in \{1,\ldots,%d\}$" % (self.n_arms))
        ax.set_ylabel("reward $R$")
        ax.set_title("Bernoulli Bandit")
        for i in range(self.n_arms):
            plt.bar(i, 0, alpha=0.6, label=f'Arm {i+1} (θ={self.means[i]:.2f})', width=0.4)
        plt.savefig("bandit_bernoulli_empty.pdf",bbox_inches='tight')

        plt.xticks(np.arange(self.n_arms), ['%d' % (_+1) for _ in range(self.n_arms)])
        ax.set_xticklabels(self.labels)
        ax.set_xlabel("action $A \in \{1,\ldots,%d\}$" % (self.n_arms))
        ax.set_ylabel("reward $R$")
        ax.set_title("Bernoulli Bandit")

        for i in range(self.n_arms):
            plt.bar(i, self.means[i], alpha=0.6, label=f'Arm {i+1} (θ={self.means[i]:.2f})', width=0.4)

        plt.savefig("bandit_bernoulli_gtruth.pdf",bbox_inches='tight')
        #plt.savefig(info['fname'],bbox_inches='tight')
        plt.show()


