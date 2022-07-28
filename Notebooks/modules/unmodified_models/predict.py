import torch
import pandas as pd
from torch.nn import Softmax
from unmodified.mod_data import ShotWayDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

def predict(model, support_set_path, query_set_path, max_query_size=20):
    """
        Parameters:
            model:
                a (pretrained) model to be used for inference
            support_set_path: str
                Path to a support set directory
            query_set_path: str
                Path to the query set directory
            max_query_size: int
                Maximum number of query points to predict at a time
        Returns:
            results: Probabilities """

    norm = Softmax(dim=1)
    output_probabilities = []
    #fnames = []
    # load the data

    support_set = ShotWayDataset(support_set_path, transform=resize, size=(255, 255))
    stacked_support_set = {}
    for s_k in set(support_set.labels):
        # get class inds
        stacked_support_set[s_k] = torch.stack([x[0] for i, x in enumerate(support_set) if x[1] == s_k])

    query_set = ShotWayDataset(query_set_path, transform=resize, size=(255, 255))
    query_loader = DataLoader(query_set, batch_size=max_query_size)
    # construct a batches to pass to the model
    # support_set is a dictionary of stacked tensors, query_set is still a Dataset object
    # batch = {'support_examples': support_set, 'query_examples': query_set}
    # model needs to switch to eval mode
    model.eval()
    with torch.no_grad():
        for i, q in enumerate(query_loader):
            # q[0] is a stack of image data q[1] can be ignored
            print("Computing batch %s" % str(i))
            output = model(stacked_support_set, q[0])
            probabilities = norm(output)
            output_probabilities.append(probabilities)
            #fnames.append(query_loader.dataset.samples[i])
    results = pd.DataFrame(tensor_i.tolist() for sublist in output_probabilities for tensor_i in sublist)
    results.columns = support_set.data.classes
    results.index = [s[0] for s in query_loader.dataset.data.samples]
    results['prediction'] = results.idxmax(axis=1)
    # TODO:make sure to flip any predictions of the support set that aren't consistent with the support set label
    return results

