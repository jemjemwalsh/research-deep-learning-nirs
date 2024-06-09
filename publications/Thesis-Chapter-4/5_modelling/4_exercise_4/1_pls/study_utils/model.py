from sklearn.cross_decomposition import PLSRegression


def create_model(
    n_components: int,
) -> PLSRegression:

    model = PLSRegression(
        n_components=n_components, 
        scale=False
    )
    return model
