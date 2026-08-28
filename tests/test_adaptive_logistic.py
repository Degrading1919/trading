from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from swingrank.adaptive.models.logistic import RidgeLogisticModel


def test_ridge_logistic_probability_and_contributions_are_exact() -> None:
    first = np.linspace(-1.0, 1.0, 80)
    second = np.sin(np.arange(80))
    features = pd.DataFrame({"first": first, "second": second})
    labels = (first + 0.15 * second > 0).astype(int)
    weights = np.ones(len(features))
    model = RidgeLogisticModel(l2_penalty=0.1).fit(features, labels, weights)

    assert model.coefficients[0] > 0
    explanation = model.explain(features)
    assert np.all((explanation.probabilities > 0) & (explanation.probabilities < 1))
    assert np.allclose(
        explanation.base_log_odds + explanation.contributions.sum(axis=1),
        explanation.log_odds,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.allclose(model.predict_proba(features), explanation.probabilities)


def test_ridge_logistic_fit_is_deterministic() -> None:
    values = np.linspace(-1.0, 1.0, 40)
    features = pd.DataFrame({"signal": values, "noise": values[::-1] ** 2})
    labels = (values > 0).astype(int)
    weights = np.linspace(0.5, 1.5, len(features))
    specification = RidgeLogisticModel(l2_penalty=0.2)

    first = specification.fit(features, labels, weights)
    second = specification.fit(features, labels, weights)

    assert first.state() == second.state()
    pdt.assert_frame_equal(first.learned_weights(), second.learned_weights())
