import models
import utils
import create_samples
import os
import multi_gpu




def train_model(data_dir, weights_file="weights.hdf5", epochs=50,batch_size=20):
    
    #create data generator
    train_data_path = os.path.join(data_dir,"train/samples")
    train_label_path = os.path.join(data_dir,"train/labels")

    val_data_path = os.path.join(data_dir, "val/samples")
    val_label_path = os.path.join(data_dir, "val/labels")

    num_train = utils.get_num_samples(train_data_path)
    num_validate = utils.get_num_samples(val_data_path)

    print(num_train)


    train_gen = create_samples.generate_samples(train_data_path,train_label_path,batch_size)
    val_gen = create_samples.generate_samples(val_data_path, val_label_path, batch_size)

    first_batch = next(train_gen)
    param_names = ["osc_wave","master_val","slave_val","xfade_val"]
    # Instantiate the model
    model, serial_model = models.setup_model(first_batch[0], param_names, weights_file=weights_file)

    save_best_only = True
    checkpointer = multi_gpu.MultiGPUModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=save_best_only,
          serial_model=serial_model, period=1, class_names=param_names)
    
    
    #train the model
    model.fit_generator(generator=train_gen,validation_data=val_gen, steps_per_epoch=num_train/batch_size, validation_steps =num_validate/batch_size )





if __name__ == '__main__':
    train_model("/Users/jacob/sound_projects/ml-synth/data")