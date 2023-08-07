##############################################################################
#
# Those few functions needed to calculate local_energy_loss
#
##############################################################################
import tensorflow as tf

def bdot(A, B): 
    """Dot product of two tensors along axis 3."""
    
    # return tensor (batch_size, 200, 200)
    return tf.reduce_sum(tf.multiply(A, B), axis=-1)

def get_magnetization(Mom):
    """Get total magnetization components."""
        
    # return tensor (batch_size, 3)
    return tf.reduce_mean(Mom, axis=(1,2))

# next nearest neighbors
def get_MR(M): return tf.roll(M, shift=-1, axis=2)  # right neighbors
def get_ML(M): return tf.roll(M, shift=1, axis=2)   # left neighbors
def get_MU(M): return tf.roll(M, shift=-1, axis=1)  # upper neighbors
def get_MD(M): return tf.roll(M, shift=1, axis=1)   # bottom neighbors


def get_batch_tensor(C):
    """Přepsal jsem tu funkci co je nahoře, protože se tensorflow nelíbí užití len(C) na symbolic tensor"""
        
    # return tensor (batch_size, 1, 1)
    output = tf.expand_dims(C, axis=(-1))
    output = tf.expand_dims(output, axis=(-1))
    return output


def get_local_energies(Mom, D, B):
    """Calculate local energies for each spin site."""
        
    # get tensors with nearest neightbors (batch_size, 200, 200, 3)
    MR, MU = get_MR(Mom), get_MU(Mom)
    ML, MD = get_ML(Mom), get_MD(Mom)
    # exchange energy (batch_size, 200, 200)
    EJ = -1. * bdot(Mom, MR + MU + ML + MD)
           
    # callculate cross products and sum over spins (batch_size, 200, 200, 3)
    def cross(M, Mi): return tf.linalg.cross(M, Mi)

    #test = (cross(Mom, MR)[:,:,:,1] - cross(Mom, MU)[:,:,:,0] - cross(Mom, ML)[:,:,:,1] + cross(Mom, MD)[:,:,:,0])
    #ED = get_batch_tensor(D) * test

    # Dzyaloshinskii-Moriya energy (batch_size, 200, 200)
    ED = get_batch_tensor(D) * (cross(Mom, MR)[:,:,:,1] - cross(Mom, MU)[:,:,:,0] - 
                                cross(Mom, ML)[:,:,:,1] + cross(Mom, MD)[:,:,:,0])
        
    # Zeeman energy (batch_size, 200, 200)
    EZ = -get_batch_tensor(B) * Mom[:,:,:,2]
    
    # return tensor (batch_size, 200, 200)
    output = EJ + ED + EZ
    return output # puvodne tnp.array(EJ + ED + EZ)




def get_local_energy_loss(Mom_true, Mom_pred, D, B):
    """Get local energy loss."""
        
    # local energies (batch_size, 200, 200)
    E_loc_true = get_local_energies(Mom_true, D, B)
    E_loc_pred = get_local_energies(Mom_pred, D, B)
    
    # return tensor (batch_size,)
    return tf.reduce_mean(
        tf.square(E_loc_true - E_loc_pred), axis=(1,2)
    )