from local_code.stage_5_code.dataset_loader import Dataset_Loader


dataset = Dataset_Loader(dName="cora", dDescription="stage 5")
dataset.dataset_source_folder_path = "./data/stage_5_data/cora"

features, labels, adj, idx_train, idx_val, idx_test = dataset.convert_to_pygcn_format(dataset.load())

print(features)
print(labels)
print(adj)
print(idx_train)
print(idx_val)
print(idx_test)



