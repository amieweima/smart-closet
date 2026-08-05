import { useEffect, useMemo, useState, useRef } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const initialForm = {
  name: "",
  category: "Top",
  colour: "",
  season: "All seasons",
  occasion: "Casual",
  warmth_level: 3,
  waterproof: "No",
  layer_type: "Base layer",

  // Kept internally so the existing backend and CSV do not break.
  style: "Casual",

  image: null,
};

function imageSource(item) {
  if (!item?.image_url) {
    return null;
  }

  return `${API_BASE_URL}${item.image_url}`;
}

function titleCase(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function broadCategoryForLabel(label) {
  const normalizedLabel = String(label).trim().toLowerCase();

  if (normalizedLabel === "hoodie") {
    return "Hoodie";
  }

  if (
    normalizedLabel === "sweater" ||
    normalizedLabel === "sweatshirt" ||
    normalizedLabel === "cardigan"
  ) {
    return "Sweater";
  }

  if (normalizedLabel === "skirt") {
    return "Skirt";
  }

  const topLabels = new Set([
    "t-shirt",
    "polo shirt",
    "button-up shirt",
    "tank top",
    "blouse",
  ]);

  const bottomLabels = new Set([
    "jeans",
    "pants",
    "sweatpants",
    "shorts",
    "cargo shorts",
  ]);

  const jacketLabels = new Set([
    "jacket",
    "windbreaker",
    "coat",
    "blazer",
  ]);

  if (topLabels.has(normalizedLabel)) {
    return "Top";
  }

  if (bottomLabels.has(normalizedLabel)) {
    return "Bottom";
  }

  if (normalizedLabel === "dress") {
    return "Dress";
  }

  if (jacketLabels.has(normalizedLabel)) {
    return "Jacket";
  }

  if (normalizedLabel === "shoes") {
    return "Shoes";
  }

  if (normalizedLabel === "baseball cap") {
    return "Accessory";
  }

  return "Other";
}

function layerTypeForLabel(label) {
  const normalizedLabel = String(label).trim().toLowerCase();

  const layerTypes = {
    "base layer": "Base layer",
    "mid layer": "Mid layer",
    "flexible top": "Flexible top",
    outerwear: "Outer layer",
    "outer layer": "Outer layer",
    "single layer": "Not a layer",
    "not a layer": "Not a layer",
  };

  return layerTypes[normalizedLabel] ?? null;
}

function styleForLabel(label) {
  const normalizedLabel = String(label).trim().toLowerCase();

  const styles = {
    casual: "Casual",
    streetwear: "Streetwear",
    "smart casual": "Smart casual",
    formal: "Formal",
    athletic: "Athletic",
    sporty: "Athletic",
    workwear: "Workwear",
    minimalist: "Minimalist",
    minimal: "Minimalist",
    vintage: "Vintage",
  };

  return styles[normalizedLabel] ?? "Other";
}

function App() {
  const [wardrobe, setWardrobe] = useState([]);
  const [form, setForm] = useState(initialForm);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [classifying, setClassifying] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [classificationAttributes, setClassificationAttributes] =
    useState(null);
  const [classificationError, setClassificationError] = useState("");

  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("All");

  const [outfit, setOutfit] = useState(null);
  const [outfitLoading, setOutfitLoading] = useState(false);
  const [outfitError, setOutfitError] = useState("");

  // === SMART CLOSET OUTFIT FEEDBACK UI ===
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const [outfitFeedback, setOutfitFeedback] = useState(null);
  const [feedbackError, setFeedbackError] = useState("");

  
  // === SMART CLOSET OCCASION WEATHER UI ===
  const [occasionPreference, setOccasionPreference] =
    useState("");
  const [weatherPreference, setWeatherPreference] =
    useState("");

// === SMART CLOSET UNIQUE OUTFITS ===
  const seenOutfitKeysRef = useRef(new Set());

  const [editingItem, setEditingItem] = useState(null);
  const [editForm, setEditForm] = useState(null);
  const [updating, setUpdating] = useState(false);
  const [editError, setEditError] = useState("");

  async function loadWardrobe() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/wardrobe`);

      if (!response.ok) {
        throw new Error("The wardrobe could not be loaded.");
      }

      const items = await response.json();

      setWardrobe(Array.isArray(items) ? items : []);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The wardrobe could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadWardrobe();
  }, []);

  function updateField(event) {
    const { name, value } = event.target;

    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function handleImageChange(event) {
    const file = event.target.files?.[0] ?? null;

    setForm((current) => ({
      ...current,
      image: file,
    }));

    setPredictions([]);
    setClassificationAttributes(null);
    setClassificationError("");
    setMessage("");
    setError("");

    if (!file) {
      return;
    }

    const body = new FormData();
    body.append("image", file);

    setClassifying(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/classify-image`,
        {
          method: "POST",
          body,
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail ?? "The image could not be classified.",
        );
      }

      const nextPredictions = Array.isArray(result.predictions)
        ? result.predictions
        : [];

      const attributes =
        result.attributes &&
          typeof result.attributes === "object"
          ? result.attributes
          : null;

      setPredictions(nextPredictions);
      setClassificationAttributes(attributes);

      const categoryLabel =
        attributes?.category?.label ??
        nextPredictions[0]?.label ??
        "";

      const colourLabel = attributes?.colour?.label ?? "";
      const layerTypeLabel =
        attributes?.layer_type?.label ?? "";
      const styleLabel = attributes?.style?.label ?? "";

      const suggestedName =
        typeof attributes?.name === "string"
          ? attributes.name
          : "";

      const suggestedSeason =
        typeof attributes?.season === "string"
          ? attributes.season
          : "";

      const suggestedOccasion =
        typeof attributes?.occasion === "string"
          ? attributes.occasion
          : "";

      const suggestedWaterproof =
        typeof attributes?.waterproof === "string"
          ? attributes.waterproof
          : "";

      const suggestedWarmth = Number(
        attributes?.warmth_level,
      );


      setForm((current) => ({
        ...current,

        category: categoryLabel
          ? broadCategoryForLabel(categoryLabel)
          : current.category,

        colour: colourLabel
          ? titleCase(colourLabel)
          : current.colour,

        layer_type:
          layerTypeForLabel(layerTypeLabel) ??
          current.layer_type,

        name: suggestedName || current.name,

        season:
          suggestedSeason || current.season,

        occasion:
          suggestedOccasion || current.occasion,

        warmth_level:
          Number.isFinite(suggestedWarmth)
            ? suggestedWarmth
            : current.warmth_level,

        waterproof:
          suggestedWaterproof || current.waterproof,

        // Style remains hidden but is still stored for compatibility.
        style: styleLabel
          ? styleForLabel(styleLabel)
          : current.style,
      }));
    } catch (requestError) {
      setClassificationError(
        requestError instanceof Error
          ? requestError.message
          : "The image could not be classified.",
      );
    } finally {
      setClassifying(false);
    }
  }

  async function submitItem(event) {
    event.preventDefault();

    const formElement = event.currentTarget;

    setError("");
    setMessage("");

    if (!form.image) {
      setError("Please choose a clothing image.");
      return;
    }

    if (!form.name.trim() || !form.colour.trim()) {
      setError("Item name and colour are required.");
      return;
    }

    const body = new FormData();

    Object.entries(form).forEach(([key, value]) => {
      body.append(key, value);
    });

    const categoryPrediction =
      classificationAttributes?.category ??
      predictions[0] ??
      null;

    const colourPrediction =
      classificationAttributes?.colour ?? null;

    const layerTypePrediction =
      classificationAttributes?.layer_type ?? null;

    const stylePrediction =
      classificationAttributes?.style ?? null;

    if (categoryPrediction?.label) {
      body.append(
        "ai_category",
        broadCategoryForLabel(categoryPrediction.label),
      );

      body.append(
        "ai_category_detail",
        categoryPrediction.label,
      );

      body.append(
        "ai_category_confidence",
        String(categoryPrediction.confidence ?? ""),
      );
    }

    if (colourPrediction?.label) {
      body.append(
        "ai_colour",
        titleCase(colourPrediction.label),
      );

      body.append(
        "ai_colour_confidence",
        String(colourPrediction.confidence ?? ""),
      );
    }

    if (layerTypePrediction?.label) {
      body.append(
        "ai_layer_type",
        layerTypeForLabel(layerTypePrediction.label) ??
        titleCase(layerTypePrediction.label),
      );

      body.append(
        "ai_layer_type_confidence",
        String(layerTypePrediction.confidence ?? ""),
      );
    }

    // Style feedback is still stored internally for compatibility.
    if (stylePrediction?.label) {
      body.append(
        "ai_style",
        styleForLabel(stylePrediction.label),
      );

      body.append(
        "ai_style_confidence",
        String(stylePrediction.confidence ?? ""),
      );
    }

    setSaving(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/wardrobe`,
        {
          method: "POST",
          body,
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail ?? "The item could not be saved.",
        );
      }

      setWardrobe((current) => [result, ...current]);
      setForm(initialForm);
      setPredictions([]);
      setClassificationAttributes(null);
      setClassificationError("");
      setOutfit(null);
      seenOutfitKeysRef.current.clear();

      formElement.reset();

      setMessage(
        `${result.name} was added to your wardrobe.`,
      );
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The item could not be saved.",
      );
    } finally {
      setSaving(false);
    }
  }



  function startEditing(item) {
    setEditingItem(item);
    setEditError("");
    setEditForm({
      name: item.name ?? "",
      category: item.category ?? "Top",
      colour: item.colour ?? "",
      season: item.season ?? "All seasons",
      occasion: item.occasion ?? "Casual",
      warmth_level: Number(item.warmth_level ?? 3),
      waterproof: item.waterproof ?? "No",
      layer_type: item.layer_type ?? "Base layer",
      style: item.style ?? "Casual",
    });
  }

  function cancelEditing() {
    if (updating) {
      return;
    }

    setEditingItem(null);
    setEditForm(null);
    setEditError("");
  }

  function updateEditField(event) {
    const { name, value } = event.target;

    setEditForm((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function saveEdit(event) {
    event.preventDefault();

    if (!editingItem || !editForm) {
      return;
    }

    if (!editForm.name.trim() || !editForm.colour.trim()) {
      setEditError("Item name and colour are required.");
      return;
    }

    setUpdating(true);
    setEditError("");
    setMessage("");
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/wardrobe/${editingItem.id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...editForm,
            warmth_level: Number(editForm.warmth_level),
          }),
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail ?? "The item could not be updated.",
        );
      }

      setWardrobe((current) =>
        current.map((item) =>
          item.id === editingItem.id
            ? result
            : item,
        ),
      );

      setOutfit(null);
      seenOutfitKeysRef.current.clear();
      setEditingItem(null);
      setEditForm(null);
      setMessage(result.message ?? `${result.name} was updated.`);
    } catch (requestError) {
      setEditError(
        requestError instanceof Error
          ? requestError.message
          : "The item could not be updated.",
      );
    } finally {
      setUpdating(false);
    }
  }

  async function deleteItem(item) {
    const confirmed = window.confirm(
      `Delete "${item.name}" from your wardrobe?`,
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/wardrobe/${item.id}`,
        {
          method: "DELETE",
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail ?? "The item could not be deleted.",
        );
      }

      setWardrobe((current) =>
        current.filter(
          (wardrobeItem) => wardrobeItem.id !== item.id,
        ),
      );

      setOutfit(null);
      seenOutfitKeysRef.current.clear();
      setMessage(result.message);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The item could not be deleted.",
      );
    }
  }

  function createOutfitKey(outfitResult) {
    const baseTop =
      outfitResult?.base_top ??
      outfitResult?.top ??
      null;

    return [
      baseTop?.id ?? "",
      outfitResult?.mid_layer?.id ?? "",
      outfitResult?.bottom?.id ?? "",
      outfitResult?.outerwear?.id ?? "",
    ].join("|");
  }

  async function generateOutfit() {
    setOutfitLoading(true);
    setOutfitError("");
    setFeedbackError("");
    setOutfitFeedback(null);

    try {
      let uniqueOutfit = null;
      let uniqueOutfitKey = "";

      // Retry because the backend currently chooses randomly.
      for (let attempt = 0; attempt < 12; attempt += 1) {
        const recommendationQuery =
          new URLSearchParams();

        if (occasionPreference) {
          recommendationQuery.set(
            "occasion",
            occasionPreference,
          );
        }

        if (weatherPreference) {
          recommendationQuery.set(
            "weather",
            weatherPreference,
          );
        }

        const queryString =
          recommendationQuery.toString();

        const recommendationUrl = queryString
          ? `${API_BASE_URL}/api/outfit-recommendation?${queryString}`
          : `${API_BASE_URL}/api/outfit-recommendation`;

        const response = await fetch(
          recommendationUrl,
        );

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.detail ??
            "An outfit could not be generated.",
          );
        }

        const resultKey = createOutfitKey(result);

        if (
          resultKey &&
          !seenOutfitKeysRef.current.has(resultKey)
        ) {
          uniqueOutfit = result;
          uniqueOutfitKey = resultKey;
          break;
        }
      }

      if (!uniqueOutfit) {
        throw new Error(
          "No new outfit combination was found. " +
          "You may have seen all currently available combinations.",
        );
      }

      seenOutfitKeysRef.current.add(uniqueOutfitKey);
      setOutfit(uniqueOutfit);
    } catch (requestError) {
      setOutfitError(
        requestError instanceof Error
          ? requestError.message
          : "An outfit could not be generated.",
      );
    } finally {
      setOutfitLoading(false);
    }
  }

  async function saveOutfitFeedback(rating) {
    if (!outfit || feedbackSaving || outfitFeedback) {
      return;
    }

    const baseTop = outfit.base_top ?? outfit.top ?? null;
    const midLayer = outfit.mid_layer ?? null;
    const bottom = outfit.bottom ?? null;
    const outerwear = outfit.outerwear ?? null;

    if (!baseTop || !bottom) {
      setFeedbackError(
        "This outfit is missing a base top or bottom.",
      );
      return;
    }

    const body = new FormData();

    body.append("rating", rating);
    body.append("base_top_id", String(baseTop.id ?? ""));
    body.append(
      "mid_layer_id",
      String(midLayer?.id ?? ""),
    );
    body.append("bottom_id", String(bottom.id ?? ""));
    body.append(
      "outerwear_id",
      String(outerwear?.id ?? ""),
    );
    body.append(
      "occasion",
      String(
        outfit?.filters?.occasion ??
        baseTop.occasion ??
        bottom.occasion ??
        midLayer?.occasion ??
        outerwear?.occasion ??
        "",
      ),
    );

    setFeedbackSaving(true);
    setFeedbackError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/outfit-feedback`,
        {
          method: "POST",
          body,
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail ??
          "The outfit feedback could not be saved.",
        );
      }

      setOutfitFeedback(result.rating);
    } catch (requestError) {
      setFeedbackError(
        requestError instanceof Error
          ? requestError.message
          : "The outfit feedback could not be saved.",
      );
    } finally {
      setFeedbackSaving(false);
    }
  }

  const categories = useMemo(() => {
    const values = new Set(
      wardrobe
        .map((item) => item.category)
        .filter(Boolean),
    );

    return ["All", ...Array.from(values).sort()];
  }, [wardrobe]);

  const visibleWardrobe = useMemo(() => {
    if (categoryFilter === "All") {
      return wardrobe;
    }

    return wardrobe.filter(
      (item) => item.category === categoryFilter,
    );
  }, [categoryFilter, wardrobe]);

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">PERSONAL WARDROBE</p>

          <h1>Smart Closet</h1>

          <p className="hero-copy">
            Organize your clothes now and build the dataset for
            smarter outfit recommendations later.
          </p>
        </div>

        <div className="item-count">
          <strong>{wardrobe.length}</strong>
          <span>items saved</span>
        </div>
      </header>

      <section className="panel outfit-panel">
        <div className="outfit-header">
          <div className="outfit-copy">
            <p className="eyebrow">OUTFIT RECOMMENDATION</p>

            <h2>Build an outfit</h2>

            <p>
              Pick the situation and let Smart Closet create a
              combination from your saved clothes.
            </p>
          </div>

          <div className="outfit-controls">
            <div className="outfit-preferences">
              <label className="outfit-field">
                <span>Occasion</span>

                <select
                  value={occasionPreference}
                  onChange={(event) => {
                    setOccasionPreference(event.target.value);
                    seenOutfitKeysRef.current.clear();
                    setOutfit(null);
                    setOutfitFeedback(null);
                  }}
                >
                  <option value="">Any occasion</option>
                  <option value="casual">Casual</option>
                  <option value="party">Party</option>
                  <option value="date">Date</option>
                  <option value="work">Work</option>
                  <option value="formal">Formal</option>
                  <option value="gym">Gym</option>
                </select>
              </label>

              <label className="outfit-field">
                <span>Weather</span>

                <select
                  value={weatherPreference}
                  onChange={(event) => {
                    setWeatherPreference(event.target.value);
                    seenOutfitKeysRef.current.clear();
                    setOutfit(null);
                    setOutfitFeedback(null);
                  }}
                >
                  <option value="">Any weather</option>
                  <option value="hot">Hot</option>
                  <option value="mild">Mild</option>
                  <option value="cold">Cold</option>
                  <option value="rainy">Rainy</option>
                </select>
              </label>
            </div>

            <button
              className="outfit-generate-button"
              type="button"
              onClick={generateOutfit}
              disabled={
                outfitLoading || wardrobe.length === 0
              }
            >
              {outfitLoading
                ? "Generating..."
                : outfit
                  ? "Generate another"
                  : "Generate outfit"}
            </button>
          </div>
        </div>

        {outfitError && (
          <p className="status error">{outfitError}</p>
        )}

        {outfit && (
          <>
            <div className="wardrobe-grid">
              {[
                [
                  outfit.roles?.base_top ?? "Base top",
                  outfit.base_top ?? outfit.top,
                ],
                [
                  outfit.roles?.mid_layer ?? "Mid layer",
                  outfit.mid_layer,
                ],
                [
                  outfit.roles?.bottom ?? "Bottom",
                  outfit.bottom,
                ],
                [
                  outfit.roles?.outerwear ?? "Outerwear",
                  outfit.outerwear,
                ],
              ]
                .filter(([, item]) => Boolean(item))
                .map(([slot, item]) => {
                  const source = imageSource(item);

                  return (
                    <article
                      className="clothing-card"
                      key={`${slot}-${item.id}`}
                    >
                      <div className="image-frame">
                        {source ? (
                          <img
                            src={source}
                            alt={item.name}
                          />
                        ) : (
                          <div className="image-placeholder">
                            Image unavailable
                          </div>
                        )}
                      </div>

                      <div className="card-content">
                        <div className="card-title-row">
                          <h3>{item.name}</h3>
                          <span>{slot}</span>
                        </div>

                        <dl>
                          <div>
                            <dt>Category</dt>
                            <dd>
                              {item.category || "Unknown"}
                            </dd>
                          </div>

                          <div>
                            <dt>Colour</dt>
                            <dd>
                              {item.colour || "Unknown"}
                            </dd>
                          </div>

                          <div>
                            <dt>Season</dt>
                            <dd>
                              {item.season || "Unknown"}
                            </dd>
                          </div>

                          <div>
                            <dt>Occasion</dt>
                            <dd>
                              {item.occasion || "Unknown"}
                            </dd>
                          </div>

                          <div>
                            <dt>Warmth</dt>
                            <dd>
                              {item.warmth_level ?? "Unknown"} / 5
                            </dd>
                          </div>
                        </dl>
                      </div>
                    </article>
                  );
                })}
            </div>

            {outfit.filters && (
              <div className="outfit-filter-summary">
                <strong>Recommendation settings:</strong>
                <span>
                  Occasion: {outfit.filters.occasion}
                </span>
                <span>
                  Weather: {outfit.filters.weather}
                </span>
              </div>
            )}

            <div className="outfit-feedback">
              <div>
                <h3>Rate this outfit</h3>
                <p>
                  Your rating will help improve future
                  recommendations.
                </p>
              </div>

              <div className="outfit-feedback-actions">
                <button
                  className="feedback-button feedback-like"
                  type="button"
                  onClick={() => saveOutfitFeedback("like")}
                  disabled={
                    feedbackSaving ||
                    Boolean(outfitFeedback)
                  }
                >
                  {feedbackSaving
                    ? "Saving..."
                    : "👍 Like"}
                </button>

                <button
                  className="feedback-button feedback-dislike"
                  type="button"
                  onClick={() =>
                    saveOutfitFeedback("dislike")
                  }
                  disabled={
                    feedbackSaving ||
                    Boolean(outfitFeedback)
                  }
                >
                  {feedbackSaving
                    ? "Saving..."
                    : "👎 Dislike"}
                </button>
              </div>

              {outfitFeedback && (
                <p className="status success">
                  Feedback saved:{" "}
                  {outfitFeedback === "like"
                    ? "You liked this outfit."
                    : "You disliked this outfit."}
                </p>
              )}

              {feedbackError && (
                <p className="status error">
                  {feedbackError}
                </p>
              )}
            </div>
          </>
        )}
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">ADD ITEM</p>
            <h2>Grow your wardrobe</h2>
          </div>
        </div>

        <form
          className="item-form"
          onSubmit={submitItem}
        >
          <label className="wide-field">
            Clothing photo

            <input
              name="image"
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={handleImageChange}
              required
            />

            {classifying && (
              <p className="status">
                Analyzing the clothing photo...
              </p>
            )}

            {classificationError && (
              <p className="status error">
                {classificationError}
              </p>
            )}

            {predictions.length > 0 && (
              <div>
                <p className="status success">
                  AI filled Category, Colour, and Layer type.
                  Review the fields below before saving.
                </p>

                <small>
                  Category detail: {predictions[0].label}{" "}
                  {predictions[0].confidence_percent}%

                  {classificationAttributes?.colour?.label
                    ? ` · Colour: ${classificationAttributes.colour.label} ${classificationAttributes.colour.confidence_percent}%`
                    : ""}

                  {classificationAttributes?.layer_type?.label
                    ? ` · Layer: ${classificationAttributes.layer_type.label}`
                    : ""}
                </small>
              </div>
            )}
          </label>

          <label>
            Item name

            <input
              name="name"
              value={form.name}
              onChange={updateField}
              placeholder="Black oversized hoodie"
              required
            />
          </label>

          <label>
            Category

            <select
              name="category"
              value={form.category}
              onChange={updateField}
            >
              {[
                "Top",
                "Hoodie",
                "Sweater",
                "Bottom",
                "Skirt",
                "Dress",
                "Jacket",
                "Shoes",
                "Accessory",
                "Other",
              ].map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            Colour

            <input
              name="colour"
              value={form.colour}
              onChange={updateField}
              placeholder="Black"
              required
            />
          </label>

          <label>
            Season

            <select
              name="season"
              value={form.season}
              onChange={updateField}
            >
              {[
                "All seasons",
                "Spring",
                "Summer",
                "Fall",
                "Winter",
              ].map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            Occasion

            <select
              name="occasion"
              value={form.occasion}
              onChange={updateField}
            >
              {[
                "Casual",
                "School",
                "Work",
                "Formal",
                "Exercise",
                "Other",
              ].map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            Warmth level

            <input
              name="warmth_level"
              type="range"
              min="1"
              max="5"
              value={form.warmth_level}
              onChange={updateField}
            />

            <span className="range-value">
              {form.warmth_level} / 5
            </span>
          </label>

          <label>
            Waterproof

            <select
              name="waterproof"
              value={form.waterproof}
              onChange={updateField}
            >
              {[
                "No",
                "Light",
                "Yes",
                "Unknown",
              ].map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            Layer type

            <select
              name="layer_type"
              value={form.layer_type}
              onChange={updateField}
            >
              {[
                "Base layer",
                "Flexible top",
                "Mid layer",
                "Outer layer",
                "Not a layer",
              ].map((option) => (
                <option
                  key={option}
                  value={option}
                >
                  {option}
                </option>
              ))}
            </select>

            <small>
              Flexible top can be worn alone or over a thin
              shirt. Hoodies and sweaters will not be layered
              over it.
            </small>
          </label>


          <div className="form-actions wide-field">
            <button
              type="submit"
              disabled={saving || classifying}
            >
              {saving
                ? "Saving..."
                : "Add to wardrobe"}
            </button>

            {error && (
              <p className="status error">{error}</p>
            )}

            {message && (
              <p className="status success">
                {message}
              </p>
            )}
          </div>
        </form>
      </section>

      <section className="wardrobe-section">
        <div className="section-heading wardrobe-heading">
          <div>
            <p className="eyebrow">MY WARDROBE</p>
            <h2>Your clothing collection</h2>
          </div>

          <label className="filter">
            Filter

            <select
              value={categoryFilter}
              onChange={(event) =>
                setCategoryFilter(event.target.value)
              }
            >
              {categories.map((category) => (
                <option
                  key={category}
                  value={category}
                >
                  {category}
                </option>
              ))}
            </select>
          </label>
        </div>

        {loading && (
          <p className="empty-state">
            Loading wardrobe...
          </p>
        )}

        {!loading &&
          error &&
          wardrobe.length === 0 && (
            <p className="empty-state">{error}</p>
          )}

        {!loading &&
          !error &&
          wardrobe.length === 0 && (
            <p className="empty-state">
              Your wardrobe is empty. Add your first clothing
              item above.
            </p>
          )}

        <div className="wardrobe-grid">
          {visibleWardrobe.map((item) => {
            const source = imageSource(item);

            return (
              <article
                className="clothing-card"
                key={item.id}
                style={{ position: "relative" }}
              >
                <button
                  type="button"
                  aria-label={`Edit ${item.name}`}
                  title={`Edit ${item.name}`}
                  onClick={() => startEditing(item)}
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "50px",
                    zIndex: 2,
                    width: "32px",
                    height: "32px",
                    padding: 0,
                    border: "none",
                    borderRadius: "50%",
                    background: "rgba(255, 255, 255, 0.94)",
                    color: "#1f1f1f",
                    fontSize: "18px",
                    fontWeight: 600,
                    lineHeight: 1,
                    display: "grid",
                    placeItems: "center",
                    cursor: "pointer",
                    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.18)",
                  }}
                >
                  ✎
                </button>

                <button
                  type="button"
                  aria-label={`Delete ${item.name}`}
                  title={`Delete ${item.name}`}
                  onClick={() => deleteItem(item)}
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    zIndex: 2,
                    width: "32px",
                    height: "32px",
                    padding: 0,
                    border: "none",
                    borderRadius: "50%",
                    background: "rgba(255, 255, 255, 0.94)",
                    color: "#1f1f1f",
                    fontSize: "22px",
                    fontWeight: 500,
                    lineHeight: 1,
                    display: "grid",
                    placeItems: "center",
                    cursor: "pointer",
                    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.18)",
                  }}
                >
                  ×
                </button>

                <div className="image-frame">
                  {source ? (
                    <img
                      src={source}
                      alt={item.name}
                    />
                  ) : (
                    <div className="image-placeholder">
                      Image unavailable
                    </div>
                  )}
                </div>

                <div className="card-content">
                  <div className="card-title-row">
                    <h3>{item.name}</h3>
                    <span>{item.category}</span>
                  </div>

                  <dl>
                    <div>
                      <dt>Colour</dt>
                      <dd>
                        {item.colour || "Unknown"}
                      </dd>
                    </div>

                    <div>
                      <dt>Season</dt>
                      <dd>
                        {item.season || "Unknown"}
                      </dd>
                    </div>

                    <div>
                      <dt>Occasion</dt>
                      <dd>
                        {item.occasion || "Unknown"}
                      </dd>
                    </div>

                    <div>
                      <dt>Warmth</dt>
                      <dd>
                        {item.warmth_level} / 5
                      </dd>
                    </div>

                    <div>
                      <dt>Waterproof</dt>
                      <dd>
                        {item.waterproof || "Unknown"}
                      </dd>
                    </div>

                    <div>
                      <dt>Layer</dt>
                      <dd>
                        {item.layer_type || "Unknown"}
                      </dd>
                    </div>
                  </dl>

                </div>
              </article>
            );
          })}
        </div>
      </section>

      {editingItem && editForm && (
        <div
          role="presentation"
          onMouseDown={cancelEditing}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            display: "grid",
            placeItems: "center",
            padding: "20px",
            background: "rgba(15, 18, 20, 0.58)",
          }}
        >
          <form
            onSubmit={saveEdit}
            onMouseDown={(event) => event.stopPropagation()}
            style={{
              width: "min(680px, 100%)",
              maxHeight: "90vh",
              overflowY: "auto",
              padding: "26px",
              borderRadius: "22px",
              background: "#ffffff",
              boxShadow: "0 24px 70px rgba(0, 0, 0, 0.28)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: "16px",
                alignItems: "center",
                marginBottom: "22px",
              }}
            >
              <div>
                <p className="eyebrow">EDIT ITEM</p>
                <h2 style={{ margin: 0 }}>
                  {editingItem.name}
                </h2>
              </div>

              <button
                type="button"
                aria-label="Close edit form"
                onClick={cancelEditing}
                disabled={updating}
                style={{
                  width: "36px",
                  height: "36px",
                  padding: 0,
                  border: "none",
                  borderRadius: "50%",
                  background: "#f0f0ec",
                  color: "#1f1f1f",
                  fontSize: "24px",
                  lineHeight: 1,
                  cursor: "pointer",
                }}
              >
                ×
              </button>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "16px",
              }}
            >
              <label>
                Item name
                <input
                  name="name"
                  value={editForm.name}
                  onChange={updateEditField}
                  required
                />
              </label>

              <label>
                Category
                <select
                  name="category"
                  value={editForm.category}
                  onChange={updateEditField}
                >
                  {[
                    "Top",
                    "Hoodie",
                    "Sweater",
                    "Bottom",
                    "Skirt",
                    "Dress",
                    "Jacket",
                    "Shoes",
                    "Accessory",
                    "Other",
                  ].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Colour
                <input
                  name="colour"
                  value={editForm.colour}
                  onChange={updateEditField}
                  required
                />
              </label>

              <label>
                Season
                <select
                  name="season"
                  value={editForm.season}
                  onChange={updateEditField}
                >
                  {[
                    "All seasons",
                    "Spring",
                    "Summer",
                    "Fall",
                    "Winter",
                  ].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Occasion
                <select
                  name="occasion"
                  value={editForm.occasion}
                  onChange={updateEditField}
                >
                  {[
                    "Casual",
                    "School",
                    "Work",
                    "Formal",
                    "Exercise",
                    "Other",
                  ].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Waterproof
                <select
                  name="waterproof"
                  value={editForm.waterproof}
                  onChange={updateEditField}
                >
                  {["No", "Light", "Yes", "Unknown"].map(
                    (option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ),
                  )}
                </select>
              </label>

              <label>
                Layer type
                <select
                  name="layer_type"
                  value={editForm.layer_type}
                  onChange={updateEditField}
                >
                  {[
                    "Base layer",
                    "Flexible top",
                    "Mid layer",
                    "Outer layer",
                    "Not a layer",
                  ].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>

                <small>
                  Flexible top can be worn alone or over a thin
                  shirt. Another hoodie or sweater will not be
                  placed over it.
                </small>
              </label>

              <label>
                Warmth level
                <input
                  name="warmth_level"
                  type="range"
                  min="1"
                  max="5"
                  value={editForm.warmth_level}
                  onChange={updateEditField}
                />
                <span className="range-value">
                  {editForm.warmth_level} / 5
                </span>
              </label>
            </div>

            {editError && (
              <p className="status error">{editError}</p>
            )}

            <div
              style={{
                display: "flex",
                justifyContent: "flex-end",
                gap: "10px",
                marginTop: "24px",
              }}
            >
              <button
                type="button"
                onClick={cancelEditing}
                disabled={updating}
                style={{
                  background: "#ecece6",
                  color: "#1f1f1f",
                }}
              >
                Cancel
              </button>

              <button type="submit" disabled={updating}>
                {updating ? "Saving..." : "Save changes"}
              </button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

export default App;