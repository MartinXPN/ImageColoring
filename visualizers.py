import matplotlib.pyplot as plt
import numpy as np

class LossVisualizer(object):
    def __init__(self, figsize=(12,8), last_iterations=5):
        self.fig, self.ax = plt.subplots(1,1)
        self.last_iterations = last_iterations


    def plot(self, fake_discriminator_loss, real_discriminator_loss, generator_loss):
        maxlen = max( len(fake_discriminator_loss), len(real_discriminator_loss), len(generator_loss) )
        while len(fake_discriminator_loss) < maxlen and fake_discriminator_loss: fake_discriminator_loss.append( fake_discriminator_loss[-1] )
        while len(real_discriminator_loss) < maxlen and fake_discriminator_loss: real_discriminator_loss.append( real_discriminator_loss[-1] )
        while len(generator_loss) < maxlen          and generator_loss:          generator_loss.append( generator_loss[-1] )
        
        fake_discriminator_loss = np.array(fake_discriminator_loss)
        real_discriminator_loss = np.array(real_discriminator_loss)
        generator_loss = np.array(generator_loss)
        
        fake_d = self.ax.plot(fake_discriminator_loss, color='red') 
        real_d = self.ax.plot(real_discriminator_loss, color='green')
        gen = self.ax.plot(generator_loss, color='blue')

        self.fig.canvas.draw()