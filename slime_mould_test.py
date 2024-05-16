import numpy as np
import math
from mealpy import IntegerVar, SMA
from PIL import Image


#Open the image, we're gonna try to get the slime mould to mimick the *image itself* as a "simple" test before we extend to edge detection.
image = Image.open("img/checker32.jpg")
image_array = np.array(image)

#How big is the image along each side? We need this in a few places below.
IMAGE_DIM=32

#Keep track of what iteration we're in. This is a bit of a hack so we can access it from within the objective function and output an image with the right name
current_iteration = 1

# Define our objective function
def objective_function(solution):
    # Use the current_iteration variable we defined globally above
    global current_iteration
    # Our slime mould parameter space is a long vector with a parameter for each pixel, so we should reshape this to the dimensions of the image
    reshaped = solution.reshape(IMAGE_DIM, IMAGE_DIM)
    # Make an image of the current slime mould position
    outimage = Image.fromarray(reshaped)
    # Save this image with the current iteration number
    outimage.convert("L").save("img/out/iteration_%05d.jpg" % current_iteration)
    # Increment the current iteration number
    current_iteration += 1
    #Work out the difference between the current slime mould position and the image we want to emulate
    differences = np.subtract(image_array, reshaped)
    # Return the sum of the squares of these differences to make this an objective function
    # (when it's large, the slime mould is very far away from the image, when its small its very close).
    return np.sum(differences ** 2)


# Set up the problem to pass to the slime mould solver.
problem_dict = {
    # Each pixel can take an integer value between 0 and 255, the number of pixels is IMAGE_DIM**2, each pixel is a paremeter
    "bounds": IntegerVar(lb=(0.,) * IMAGE_DIM**2, ub=(255.,) * IMAGE_DIM**2),
    # We want to *minimise* rather than maximise the objective function, as it gives a measure of our error
    "minmax": "min",
    # And the objective function we defined above.
    "obj_func": objective_function
}

# We create the slime mould solver, these parameters i have set pretty arbitrarily, we can look into them later.
# The number of iterations that actually get run are epoch*pop_size, so in this case 5000.
model = SMA.DevSMA(epoch=2000, pop_size=5, p_t = 0.03)
# And we pass the problem dictionary we set up above and start solving.
g_best = model.solve(problem_dict)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
