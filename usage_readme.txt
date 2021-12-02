## Preleminary
Create a folder named data in the ocrtoc_perception directory.

----------------------------

After going into Docker with exec_container.sh for each terminal, do following:

## Terminal 1
Run simulation normally.

## Terminal 2

### Option 1
If you'd like to save the data from perception. Run standart launch file:
    roslaunch ocrtoc_task solution.launch

### Option 2    
If you'd like to rerun old perception result for that particular scene (Code expects you to have the data at its folder, otherwise it will give error.), run:
    roslaunch ocrtoc_task solution_perception_reuse.launch
    
## Terminal 3
Same as terminal 1, just run original command, e.g., :
    roslaunch ocrtoc_task trigger.launch task_index:=0-0
    
    
## Alternative usage with Jupyter Notebook for data analysis and filtering:

Go into directory OCRTOC_software_package/ocrtoc_perception/src/ocrtoc_perception. In there run:
    jupyter notebook --allow-root

Then go into the link it shows in the terminal. Open grasp_visualize.ipynb. It's intuitive, just run all cells and you'll see what happens.






