// Application State
const appState = {
    currentTab: 'detection',
    detectedIngredients: [],
    confirmedIngredients: [],
    preferences: {
        flavor: null,
        flavorIntensity: 5,
        textures: [],
        dietary: 'none',
        cookingTime: 30,
        servingSize: 4
    },
    currentRecipe: null,
    cookingState: {
        currentStep: 0,
        totalSteps: 0,
        timerActive: false,
        timerDuration: 0,
        timerRemaining: 0,
        timerInterval: null
    },
    feedback: {
        history: []
    }
};

// Sample Recipe Data
const sampleRecipes = [
    {
        "id": 1,
        "title": "Spicy Chicken Stir-Fry",
        "description": "A flavorful stir-fry with tender chicken and crisp vegetables in a spicy sauce",
        "cookTime": 20,
        "servings": 4,
        "difficulty": "Easy",
        "ingredients": [
            {"item": "Chicken breast", "amount": "1 lb", "prep": "cut into strips"},
            {"item": "Bell peppers", "amount": "2", "prep": "sliced"},
            {"item": "Onion", "amount": "1 medium", "prep": "sliced"},
            {"item": "Garlic", "amount": "3 cloves", "prep": "minced"},
            {"item": "Soy sauce", "amount": "3 tbsp", "prep": ""},
            {"item": "Sriracha", "amount": "2 tbsp", "prep": ""},
            {"item": "Olive oil", "amount": "2 tbsp", "prep": ""},
            {"item": "Rice", "amount": "2 cups cooked", "prep": ""}
        ],
        "steps": [
            {"step": 1, "instruction": "Heat olive oil in a large pan over medium-high heat", "timing": 2, "type": "prep"},
            {"step": 2, "instruction": "Cook chicken strips until golden brown and cooked through", "timing": 8, "type": "cooking"},
            {"step": 3, "instruction": "Add garlic and cook until fragrant", "timing": 1, "type": "cooking"},
            {"step": 4, "instruction": "Add bell peppers and onions, stir-fry until tender-crisp", "timing": 5, "type": "cooking"},
            {"step": 5, "instruction": "Mix soy sauce and sriracha in a small bowl", "timing": 1, "type": "prep"},
            {"step": 6, "instruction": "Pour sauce over chicken and vegetables, toss to coat", "timing": 2, "type": "cooking"},
            {"step": 7, "instruction": "Let flavors meld and sauce thicken slightly", "timing": 2, "type": "cooking"},
            {"step": 8, "instruction": "Serve hot over cooked rice", "timing": 0, "type": "serving"}
        ],
        "nutrition": {"calories": 380, "protein": "28g", "carbs": "35g", "fat": "12g"}
    },
    {
        "id": 2,
        "title": "Creamy Mushroom Risotto",
        "description": "Rich and creamy risotto with sautéed mushrooms and fresh herbs",
        "cookTime": 35,
        "servings": 4,
        "difficulty": "Medium",
        "ingredients": [
            {"item": "Arborio rice", "amount": "1.5 cups", "prep": ""},
            {"item": "Mushrooms", "amount": "8 oz", "prep": "sliced"},
            {"item": "Onion", "amount": "1 small", "prep": "finely diced"},
            {"item": "Garlic", "amount": "2 cloves", "prep": "minced"},
            {"item": "White wine", "amount": "1/2 cup", "prep": ""},
            {"item": "Chicken stock", "amount": "4 cups", "prep": "warm"},
            {"item": "Butter", "amount": "3 tbsp", "prep": ""},
            {"item": "Parmesan cheese", "amount": "1/2 cup", "prep": "grated"}
        ],
        "steps": [
            {"step": 1, "instruction": "Heat stock in a separate pan and keep warm", "timing": 0, "type": "prep"},
            {"step": 2, "instruction": "Sauté mushrooms in 1 tbsp butter until golden, set aside", "timing": 5, "type": "prep"},
            {"step": 3, "instruction": "In same pan, cook onion and garlic until softened", "timing": 3, "type": "cooking"},
            {"step": 4, "instruction": "Add rice and stir to coat with oil, cook until edges are translucent", "timing": 2, "type": "cooking"},
            {"step": 5, "instruction": "Add wine and stir until absorbed", "timing": 2, "type": "cooking"},
            {"step": 6, "instruction": "Add stock one ladle at a time, stirring continuously until absorbed", "timing": 18, "type": "cooking"},
            {"step": 7, "instruction": "Stir in mushrooms, remaining butter, and Parmesan", "timing": 2, "type": "finishing"},
            {"step": 8, "instruction": "Season with salt and pepper, serve immediately", "timing": 0, "type": "serving"}
        ],
        "nutrition": {"calories": 420, "protein": "12g", "carbs": "65g", "fat": "14g"}
    },
    {
        "id": 3,
        "title": "Crispy Fish Tacos",
        "description": "Light and crispy fish tacos with fresh slaw and zesty lime crema",
        "cookTime": 25,
        "servings": 6,
        "difficulty": "Easy",
        "ingredients": [
            {"item": "White fish fillets", "amount": "1.5 lbs", "prep": "cut into strips"},
            {"item": "Corn tortillas", "amount": "12", "prep": ""},
            {"item": "Cabbage", "amount": "2 cups", "prep": "shredded"},
            {"item": "Lime", "amount": "2", "prep": "juiced"},
            {"item": "Sour cream", "amount": "1/2 cup", "prep": ""},
            {"item": "Flour", "amount": "1 cup", "prep": ""},
            {"item": "Beer", "amount": "3/4 cup", "prep": "cold"},
            {"item": "Oil for frying", "amount": "2 cups", "prep": ""}
        ],
        "steps": [
            {"step": 1, "instruction": "Heat oil to 375°F in a large pot", "timing": 5, "type": "prep"},
            {"step": 2, "instruction": "Mix flour and beer to create smooth batter", "timing": 2, "type": "prep"},
            {"step": 3, "instruction": "Combine cabbage with lime juice for slaw", "timing": 3, "type": "prep"},
            {"step": 4, "instruction": "Mix sour cream with remaining lime juice for crema", "timing": 1, "type": "prep"},
            {"step": 5, "instruction": "Dip fish in batter and fry until golden and crispy", "timing": 8, "type": "cooking"},
            {"step": 6, "instruction": "Drain fish on paper towels", "timing": 2, "type": "cooking"},
            {"step": 7, "instruction": "Warm tortillas in dry pan or microwave", "timing": 2, "type": "prep"},
            {"step": 8, "instruction": "Assemble tacos with fish, slaw, and crema", "timing": 0, "type": "serving"}
        ],
        "nutrition": {"calories": 340, "protein": "25g", "carbs": "28g", "fat": "15g"}
    }
];

const sampleIngredients = ["tomatoes", "onions", "garlic", "chicken breast", "olive oil", "bell peppers", "rice", "mushrooms", "lime", "cilantro"];

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    // Small delay to ensure all elements are rendered
    setTimeout(() => {
        initializeApp();
    }, 100);
});

function initializeApp() {
    console.log('Setting up components...');
    setupTabNavigation();
    setupIngredientDetection();
    setupPreferences();
    setupRecipeGeneration();
    setupCookingMode();
    setupFeedback();
    
    // Ensure we start with detection tab visible
    switchTab('detection');
    
    loadSessionState();
    console.log('App initialized successfully');
}

// Tab Navigation - Fixed version
function setupTabNavigation() {
    console.log('Setting up tab navigation...');
    const tabButtons = document.querySelectorAll('.tab-btn');
    console.log('Found tab buttons:', tabButtons.length);

    tabButtons.forEach((button, index) => {
        console.log(`Setting up button ${index}:`, button.dataset.tab);
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const targetTab = button.dataset.tab;
            console.log('Tab clicked:', targetTab);
            switchTab(targetTab);
        });
    });
}

function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    // Update button states
    const allButtons = document.querySelectorAll('.tab-btn');
    allButtons.forEach(btn => {
        btn.classList.remove('active');
        console.log('Removed active from:', btn.dataset.tab);
    });
    
    const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeButton) {
        activeButton.classList.add('active');
        console.log('Added active to button:', tabName);
    } else {
        console.error('Active button not found for:', tabName);
    }

    // Update panel visibility - this is the critical fix
    const allPanels = document.querySelectorAll('.tab-panel');
    console.log('Found panels:', allPanels.length);
    
    allPanels.forEach(panel => {
        panel.style.display = 'none';
        panel.classList.add('hidden');
        console.log('Hidden panel:', panel.id);
    });
    
    const activePanel = document.getElementById(`${tabName}-tab`);
    if (activePanel) {
        activePanel.style.display = 'block';
        activePanel.classList.remove('hidden');
        console.log('Shown panel:', activePanel.id);
    } else {
        console.error('Active panel not found for:', `${tabName}-tab`);
    }

    appState.currentTab = tabName;
    saveSessionState();
    
    console.log('Tab switch completed for:', tabName);
}

// Ingredient Detection
function setupIngredientDetection() {
    console.log('Setting up ingredient detection...');
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageUpload');
    const confirmBtn = document.getElementById('confirmIngredients');
    const redetectBtn = document.getElementById('redetect');

    if (!uploadArea || !fileInput) {
        console.error('Upload elements not found');
        return;
    }

    // File upload handling
    uploadArea.addEventListener('click', () => {
        console.log('Upload area clicked');
        fileInput.click();
    });
    
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Button handlers
    if (confirmBtn) {
        confirmBtn.addEventListener('click', confirmIngredients);
    }
    if (redetectBtn) {
        redetectBtn.addEventListener('click', redetectIngredients);
    }
    
    console.log('Ingredient detection setup complete');
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processImageFile(files[0]);
    }
}

function handleFileSelect(e) {
    console.log('File selected');
    const file = e.target.files[0];
    if (file) {
        processImageFile(file);
    }
}

function processImageFile(file) {
    console.log('Processing image file:', file.name);
    
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    // Show processing animation
    const resultsDiv = document.getElementById('detectionResults');
    const processingDiv = document.getElementById('aiProcessing');
    
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
        resultsDiv.style.display = 'block';
    }
    if (processingDiv) {
        processingDiv.classList.remove('hidden');
        processingDiv.style.display = 'flex';
    }
    
    // Simulate AI processing delay
    setTimeout(() => {
        if (processingDiv) {
            processingDiv.classList.add('hidden');
            processingDiv.style.display = 'none';
        }
        mockIngredientDetection();
    }, 2000);
}

function mockIngredientDetection() {
    console.log('Running mock ingredient detection...');
    
    // Generate mock detected ingredients
    const numIngredients = Math.floor(Math.random() * 4) + 4; // 4-7 ingredients
    const detectedIngredients = [];
    const shuffled = [...sampleIngredients].sort(() => 0.5 - Math.random());
    
    for (let i = 0; i < numIngredients; i++) {
        detectedIngredients.push({
            name: shuffled[i],
            confidence: Math.floor(Math.random() * 20) + 80 // 80-99%
        });
    }
    
    appState.detectedIngredients = detectedIngredients;
    console.log('Detected ingredients:', detectedIngredients);
    displayDetectedIngredients();
}

function displayDetectedIngredients() {
    console.log('Displaying detected ingredients...');
    const ingredientsList = document.getElementById('ingredientsList');
    
    if (!ingredientsList) {
        console.error('Ingredients list element not found');
        return;
    }
    
    ingredientsList.innerHTML = '';
    
    appState.detectedIngredients.forEach((ingredient, index) => {
        const item = document.createElement('div');
        item.className = 'ingredient-item fade-in';
        item.innerHTML = `
            <div class="ingredient-info">
                <input type="checkbox" class="ingredient-checkbox" id="ingredient-${index}" checked>
                <label for="ingredient-${index}" class="ingredient-name">${ingredient.name}</label>
            </div>
            <div class="confidence-score">${ingredient.confidence}%</div>
        `;
        ingredientsList.appendChild(item);
    });
    
    console.log('Ingredients displayed successfully');
}

function confirmIngredients() {
    console.log('Confirming ingredients...');
    const checkedBoxes = document.querySelectorAll('.ingredient-checkbox:checked');
    appState.confirmedIngredients = [];
    
    checkedBoxes.forEach(checkbox => {
        const index = parseInt(checkbox.id.split('-')[1]);
        if (appState.detectedIngredients[index]) {
            appState.confirmedIngredients.push(appState.detectedIngredients[index].name);
        }
    });
    
    if (appState.confirmedIngredients.length === 0) {
        alert('Please select at least one ingredient.');
        return;
    }
    
    console.log('Confirmed ingredients:', appState.confirmedIngredients);
    
    // Show confirmation and switch to preferences
    showNotification('Ingredients confirmed! Set your preferences next.', 'success');
    setTimeout(() => switchTab('preferences'), 1000);
    saveSessionState();
}

function redetectIngredients() {
    console.log('Re-detecting ingredients...');
    const processingDiv = document.getElementById('aiProcessing');
    
    if (processingDiv) {
        processingDiv.classList.remove('hidden');
        processingDiv.style.display = 'flex';
    }
    
    setTimeout(() => {
        if (processingDiv) {
            processingDiv.classList.add('hidden');
            processingDiv.style.display = 'none';
        }
        mockIngredientDetection();
    }, 1500);
}

// Preferences Setup
function setupPreferences() {
    console.log('Setting up preferences...');
    setupFlavorPreferences();
    setupTexturePreferences();
    setupAdditionalPreferences();
    
    const saveBtn = document.getElementById('savePreferences');
    if (saveBtn) {
        saveBtn.addEventListener('click', savePreferences);
    }
}

function setupFlavorPreferences() {
    const flavorInputs = document.querySelectorAll('input[name="flavor"]');
    const intensitySliders = document.querySelectorAll('.slider[data-flavor]');
    
    flavorInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            appState.preferences.flavor = e.target.value;
            console.log('Flavor selected:', e.target.value);
        });
    });
    
    intensitySliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const flavor = e.target.dataset.flavor;
            const value = e.target.value;
            const valueDisplay = e.target.parentElement.querySelector('.intensity-value');
            if (valueDisplay) {
                valueDisplay.textContent = value;
            }
            
            if (appState.preferences.flavor === flavor) {
                appState.preferences.flavorIntensity = parseInt(value);
            }
        });
    });
}

function setupTexturePreferences() {
    const textureCheckboxes = document.querySelectorAll('.texture-checkbox input[type="checkbox"]');
    
    textureCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            appState.preferences.textures = [];
            document.querySelectorAll('.texture-checkbox input:checked').forEach(cb => {
                appState.preferences.textures.push(cb.value);
            });
            console.log('Textures selected:', appState.preferences.textures);
        });
    });
}

function setupAdditionalPreferences() {
    const dietarySelect = document.getElementById('dietaryRestrictions');
    const cookingTimeSlider = document.getElementById('cookingTime');
    const servingSizeInput = document.getElementById('servingSize');
    
    if (dietarySelect) {
        dietarySelect.addEventListener('change', (e) => {
            appState.preferences.dietary = e.target.value;
        });
    }
    
    if (cookingTimeSlider) {
        cookingTimeSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            appState.preferences.cookingTime = parseInt(value);
            const timeValue = document.querySelector('.time-value');
            if (timeValue) {
                timeValue.textContent = value > 60 ? '2+ hours' : `${value} min`;
            }
        });
    }
    
    if (servingSizeInput) {
        servingSizeInput.addEventListener('change', (e) => {
            appState.preferences.servingSize = parseInt(e.target.value);
        });
    }
}

function savePreferences() {
    console.log('Saving preferences...');
    
    if (!appState.preferences.flavor) {
        alert('Please select a flavor preference.');
        return;
    }
    
    if (appState.preferences.textures.length === 0) {
        alert('Please select at least one texture preference.');
        return;
    }
    
    const confirmation = document.getElementById('preferencesConfirmation');
    if (confirmation) {
        confirmation.classList.remove('hidden');
        setTimeout(() => confirmation.classList.add('hidden'), 3000);
    }
    
    showNotification('Preferences saved successfully!', 'success');
    saveSessionState();
}

// Recipe Generation
function setupRecipeGeneration() {
    console.log('Setting up recipe generation...');
    
    const generateBtn = document.getElementById('generateRecipe');
    const startBtn = document.getElementById('startCooking');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateRecipe);
    }
    if (startBtn) {
        startBtn.addEventListener('click', startCooking);
    }
}

function generateRecipe() {
    console.log('Generating recipe...');
    
    if (appState.confirmedIngredients.length === 0) {
        alert('Please detect and confirm ingredients first.');
        switchTab('detection');
        return;
    }
    
    if (!appState.preferences.flavor) {
        alert('Please set your preferences first.');
        switchTab('preferences');
        return;
    }
    
    // Show processing
    const resultsDiv = document.getElementById('recipeResults');
    const processingDiv = document.getElementById('recipeProcessing');
    
    if (resultsDiv) {
        resultsDiv.classList.remove('hidden');
        resultsDiv.style.display = 'block';
    }
    if (processingDiv) {
        processingDiv.classList.remove('hidden');
        processingDiv.style.display = 'flex';
    }
    
    // Simulate AI processing
    setTimeout(() => {
        if (processingDiv) {
            processingDiv.classList.add('hidden');
            processingDiv.style.display = 'none';
        }
        selectAndDisplayRecipe();
    }, 3000);
}

function selectAndDisplayRecipe() {
    console.log('Selecting recipe based on preferences...');
    
    // Select recipe based on preferences (simplified matching)
    let selectedRecipe;
    
    if (appState.preferences.flavor === 'spicy') {
        selectedRecipe = sampleRecipes[0]; // Spicy Chicken Stir-Fry
    } else if (appState.preferences.textures.includes('creamy')) {
        selectedRecipe = sampleRecipes[1]; // Creamy Mushroom Risotto
    } else if (appState.preferences.textures.includes('crispy')) {
        selectedRecipe = sampleRecipes[2]; // Crispy Fish Tacos
    } else {
        selectedRecipe = sampleRecipes[Math.floor(Math.random() * sampleRecipes.length)];
    }
    
    appState.currentRecipe = selectedRecipe;
    console.log('Selected recipe:', selectedRecipe.title);
    displayRecipe(selectedRecipe);
}

function displayRecipe(recipe) {
    console.log('Displaying recipe:', recipe.title);
    
    // Set recipe details
    const titleElement = document.getElementById('recipeTitle');
    const descElement = document.getElementById('recipeDescription');
    const cookTimeElement = document.getElementById('recipeCookTime');
    const servingsElement = document.getElementById('recipeServings');
    const difficultyElement = document.getElementById('recipeDifficulty');
    
    if (titleElement) titleElement.textContent = recipe.title;
    if (descElement) descElement.textContent = recipe.description;
    if (cookTimeElement) cookTimeElement.textContent = recipe.cookTime;
    if (servingsElement) servingsElement.textContent = recipe.servings;
    if (difficultyElement) difficultyElement.textContent = recipe.difficulty;
    
    // Ingredients
    const ingredientsList = document.getElementById('recipeIngredients');
    if (ingredientsList) {
        ingredientsList.innerHTML = '';
        recipe.ingredients.forEach(ingredient => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${ingredient.amount} ${ingredient.item}</span>
                <span class="prep">${ingredient.prep}</span>
            `;
            ingredientsList.appendChild(li);
        });
    }
    
    // Nutrition
    const nutritionDiv = document.getElementById('nutritionInfo');
    if (nutritionDiv) {
        nutritionDiv.innerHTML = '';
        Object.entries(recipe.nutrition).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'nutrition-item';
            item.innerHTML = `
                <div class="nutrition-value">${value}</div>
                <div class="nutrition-label">${key}</div>
            `;
            nutritionDiv.appendChild(item);
        });
    }
    
    const recipeDisplay = document.getElementById('recipeDisplay');
    if (recipeDisplay) {
        recipeDisplay.classList.remove('hidden');
        recipeDisplay.style.display = 'block';
    }
    
    saveSessionState();
}

function startCooking() {
    console.log('Starting cooking mode...');
    
    if (!appState.currentRecipe) {
        alert('No recipe selected.');
        return;
    }
    
    // Initialize cooking state
    appState.cookingState = {
        currentStep: 0,
        totalSteps: appState.currentRecipe.steps.length,
        timerActive: false,
        timerDuration: 0,
        timerRemaining: 0,
        timerInterval: null
    };
    
    switchTab('cooking');
    setTimeout(() => initializeCookingInterface(), 100);
}

// Cooking Mode
function setupCookingMode() {
    console.log('Setting up cooking mode...');
    
    const nextBtn = document.getElementById('nextStep');
    const prevBtn = document.getElementById('prevStep');
    const skipBtn = document.getElementById('skipTimer');
    const finishBtn = document.getElementById('finishCooking');
    const goToRecipeBtn = document.getElementById('goToRecipe');
    
    if (nextBtn) nextBtn.addEventListener('click', nextStep);
    if (prevBtn) prevBtn.addEventListener('click', prevStep);
    if (skipBtn) skipBtn.addEventListener('click', skipTimer);
    if (finishBtn) finishBtn.addEventListener('click', finishCooking);
    if (goToRecipeBtn) goToRecipeBtn.addEventListener('click', () => switchTab('recipe'));
}

function initializeCookingInterface() {
    console.log('Initializing cooking interface...');
    
    if (!appState.currentRecipe) {
        const noCookingMode = document.getElementById('noCookingMode');
        const cookingInterface = document.getElementById('cookingInterface');
        
        if (noCookingMode) noCookingMode.style.display = 'block';
        if (cookingInterface) cookingInterface.classList.add('hidden');
        return;
    }
    
    const noCookingMode = document.getElementById('noCookingMode');
    const cookingInterface = document.getElementById('cookingInterface');
    const cookingComplete = document.getElementById('cookingComplete');
    
    if (noCookingMode) noCookingMode.style.display = 'none';
    if (cookingInterface) {
        cookingInterface.classList.remove('hidden');
        cookingInterface.style.display = 'block';
    }
    if (cookingComplete) cookingComplete.classList.add('hidden');
    
    displayCurrentStep();
}

function displayCurrentStep() {
    console.log('Displaying step:', appState.cookingState.currentStep + 1);
    
    const step = appState.currentRecipe.steps[appState.cookingState.currentStep];
    const stepNumber = appState.cookingState.currentStep + 1;
    
    const stepTitle = document.getElementById('stepTitle');
    const stepInstruction = document.getElementById('stepInstruction');
    
    if (stepTitle) stepTitle.textContent = `Step ${stepNumber}`;
    if (stepInstruction) stepInstruction.textContent = step.instruction;
    
    // Update progress
    const progress = (stepNumber / appState.cookingState.totalSteps) * 100;
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressText) progressText.textContent = `Step ${stepNumber} of ${appState.cookingState.totalSteps}`;
    
    // Handle timer
    if (step.timing > 0) {
        setupStepTimer(step.timing);
    } else {
        hideTimer();
        enableNextStep();
    }
    
    // Update navigation buttons
    const prevBtn = document.getElementById('prevStep');
    if (prevBtn) {
        prevBtn.disabled = appState.cookingState.currentStep === 0;
    }
    
    saveSessionState();
}

function setupStepTimer(minutes) {
    console.log('Setting up timer for', minutes, 'minutes');
    
    const timerSection = document.getElementById('timerSection');
    const timerText = document.getElementById('timerText');
    const timerFill = document.getElementById('timerFill');
    const timerStatus = document.getElementById('timerStatus');
    
    if (timerSection) {
        timerSection.classList.remove('hidden');
        timerSection.style.display = 'block';
    }
    
    appState.cookingState.timerDuration = minutes * 60; // Convert to seconds
    appState.cookingState.timerRemaining = appState.cookingState.timerDuration;
    appState.cookingState.timerActive = true;
    
    if (timerStatus) {
        timerStatus.textContent = `Timer started for ${minutes} minute${minutes > 1 ? 's' : ''}`;
    }
    
    // Disable next button until timer completes
    disableNextStep();
    
    // Start countdown
    appState.cookingState.timerInterval = setInterval(() => {
        appState.cookingState.timerRemaining--;
        
        const mins = Math.floor(appState.cookingState.timerRemaining / 60);
        const secs = appState.cookingState.timerRemaining % 60;
        
        if (timerText) {
            timerText.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        
        // Update progress bar
        const progress = ((appState.cookingState.timerDuration - appState.cookingState.timerRemaining) / appState.cookingState.timerDuration) * 100;
        if (timerFill) {
            timerFill.style.width = `${progress}%`;
        }
        
        if (appState.cookingState.timerRemaining <= 0) {
            timerComplete();
        }
    }, 1000);
}

function timerComplete() {
    console.log('Timer completed');
    
    if (appState.cookingState.timerInterval) {
        clearInterval(appState.cookingState.timerInterval);
    }
    appState.cookingState.timerActive = false;
    
    const timerSection = document.getElementById('timerSection');
    const timerStatus = document.getElementById('timerStatus');
    
    if (timerSection) timerSection.classList.add('timer-complete');
    if (timerStatus) timerStatus.textContent = '✅ Timer complete! Ready for next step.';
    
    // Simple alert notification
    alert('🔔 Timer complete! Ready for next step.');
    
    enableNextStep();
}

function hideTimer() {
    const timerSection = document.getElementById('timerSection');
    if (timerSection) {
        timerSection.classList.add('hidden');
        timerSection.style.display = 'none';
    }
}

function enableNextStep() {
    const nextBtn = document.getElementById('nextStep');
    if (nextBtn) {
        nextBtn.disabled = false;
    }
}

function disableNextStep() {
    const nextBtn = document.getElementById('nextStep');
    if (nextBtn) {
        nextBtn.disabled = true;
    }
}

function nextStep() {
    console.log('Moving to next step');
    
    if (appState.cookingState.currentStep < appState.cookingState.totalSteps - 1) {
        appState.cookingState.currentStep++;
        clearCurrentTimer();
        displayCurrentStep();
    } else {
        // Cooking complete
        showCookingComplete();
    }
}

function prevStep() {
    console.log('Moving to previous step');
    
    if (appState.cookingState.currentStep > 0) {
        appState.cookingState.currentStep--;
        clearCurrentTimer();
        displayCurrentStep();
    }
}

function skipTimer() {
    console.log('Skipping timer');
    
    if (appState.cookingState.timerActive) {
        if (appState.cookingState.timerInterval) {
            clearInterval(appState.cookingState.timerInterval);
        }
        appState.cookingState.timerActive = false;
        
        const timerStatus = document.getElementById('timerStatus');
        if (timerStatus) {
            timerStatus.textContent = '⏭️ Timer skipped (debug mode)';
        }
        enableNextStep();
    }
}

function clearCurrentTimer() {
    if (appState.cookingState.timerInterval) {
        clearInterval(appState.cookingState.timerInterval);
    }
    appState.cookingState.timerActive = false;
    
    const timerSection = document.getElementById('timerSection');
    if (timerSection) {
        timerSection.classList.remove('timer-complete');
    }
}

function showCookingComplete() {
    console.log('Cooking complete');
    
    const cookingComplete = document.getElementById('cookingComplete');
    if (cookingComplete) {
        cookingComplete.classList.remove('hidden');
        cookingComplete.style.display = 'block';
    }
    hideTimer();
}

function finishCooking() {
    switchTab('feedback');
}

// Feedback System
function setupFeedback() {
    console.log('Setting up feedback...');
    
    const correctBtn = document.getElementById('ingredientsCorrect');
    const incorrectBtn = document.getElementById('ingredientsIncorrect');
    const submitBtn = document.getElementById('submitFeedback');
    const toggleBtn = document.getElementById('toggleHistory');
    
    if (correctBtn) correctBtn.addEventListener('click', () => recordIngredientFeedback(true));
    if (incorrectBtn) incorrectBtn.addEventListener('click', () => recordIngredientFeedback(false));
    if (submitBtn) submitBtn.addEventListener('click', submitFeedback);
    if (toggleBtn) toggleBtn.addEventListener('click', toggleFeedbackHistory);
    
    setupStarRatings();
}

function setupStarRatings() {
    const starRatings = document.querySelectorAll('.star-rating');
    
    starRatings.forEach(rating => {
        const stars = rating.querySelectorAll('.star');
        let currentRating = 0;
        
        stars.forEach((star, index) => {
            star.addEventListener('click', () => {
                currentRating = index + 1;
                updateStarDisplay(stars, currentRating);
                rating.dataset.currentRating = currentRating;
            });
            
            star.addEventListener('mouseover', () => {
                updateStarDisplay(stars, index + 1);
            });
        });
        
        rating.addEventListener('mouseleave', () => {
            updateStarDisplay(stars, currentRating);
        });
    });
}

function updateStarDisplay(stars, rating) {
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
}

function recordIngredientFeedback(correct) {
    const feedback = {
        type: 'ingredient_detection',
        correct: correct,
        timestamp: new Date().toISOString()
    };
    
    appState.feedback.history.push(feedback);
    
    const message = correct ? 
        '✅ Thank you! This helps improve our AI detection.' : 
        '❌ Thanks for the feedback! We\'ll work on better detection.';
    
    showNotification(message, 'info');
    saveSessionState();
}

function submitFeedback() {
    console.log('Submitting feedback...');
    
    const feedback = {
        type: 'recipe_quality',
        timestamp: new Date().toISOString(),
        ratings: {},
        corrections: '',
        detailedFeedback: '',
        recipeId: appState.currentRecipe?.id
    };
    
    // Get form values
    const correctionsField = document.getElementById('ingredientCorrections');
    const detailedField = document.getElementById('detailedFeedback');
    
    if (correctionsField) feedback.corrections = correctionsField.value;
    if (detailedField) feedback.detailedFeedback = detailedField.value;
    
    // Collect star ratings
    document.querySelectorAll('.star-rating').forEach(rating => {
        const ratingType = rating.dataset.rating;
        const currentRating = rating.dataset.currentRating || 0;
        feedback.ratings[ratingType] = parseInt(currentRating);
    });
    
    appState.feedback.history.push(feedback);
    
    // Show confirmation
    const confirmation = document.getElementById('feedbackConfirmation');
    if (confirmation) {
        confirmation.classList.remove('hidden');
        setTimeout(() => confirmation.classList.add('hidden'), 3000);
    }
    
    // Update learning progress (mock)
    updateLearningProgress();
    
    // Clear form
    document.querySelectorAll('.star-rating').forEach(rating => {
        rating.dataset.currentRating = 0;
        const stars = rating.querySelectorAll('.star');
        updateStarDisplay(stars, 0);
    });
    
    if (correctionsField) correctionsField.value = '';
    if (detailedField) detailedField.value = '';
    
    saveSessionState();
    showNotification('Feedback submitted successfully!', 'success');
}

function updateLearningProgress() {
    const learningProgress = document.getElementById('learningProgress');
    const personalizationScore = document.getElementById('personalizationScore');
    
    if (learningProgress && personalizationScore) {
        // Simulate progress increase
        const currentLearning = parseInt(learningProgress.style.width) || 67;
        const currentPersonalization = parseInt(personalizationScore.style.width) || 84;
        
        const newLearning = Math.min(100, currentLearning + Math.floor(Math.random() * 5) + 1);
        const newPersonalization = Math.min(100, currentPersonalization + Math.floor(Math.random() * 3) + 1);
        
        learningProgress.style.width = `${newLearning}%`;
        personalizationScore.style.width = `${newPersonalization}%`;
        
        const learningText = learningProgress.nextElementSibling;
        const personalizationText = personalizationScore.nextElementSibling;
        
        if (learningText) learningText.textContent = `${newLearning}%`;
        if (personalizationText) personalizationText.textContent = `${newPersonalization}%`;
    }
}

function toggleFeedbackHistory() {
    const historyContent = document.getElementById('historyContent');
    const toggleBtn = document.getElementById('toggleHistory');
    
    if (historyContent && toggleBtn) {
        if (historyContent.classList.contains('hidden')) {
            displayFeedbackHistory();
            historyContent.classList.remove('hidden');
            historyContent.style.display = 'block';
            toggleBtn.textContent = 'Hide History';
        } else {
            historyContent.classList.add('hidden');
            historyContent.style.display = 'none';
            toggleBtn.textContent = 'Show History';
        }
    }
}

function displayFeedbackHistory() {
    const historyContent = document.getElementById('historyContent');
    
    if (!historyContent) return;
    
    if (appState.feedback.history.length === 0) {
        historyContent.innerHTML = '<p>No feedback history yet. Start cooking to build your personalization data!</p>';
        return;
    }
    
    let historyHTML = '<ul>';
    appState.feedback.history.forEach((feedback, index) => {
        const date = new Date(feedback.timestamp).toLocaleDateString();
        const time = new Date(feedback.timestamp).toLocaleTimeString();
        
        if (feedback.type === 'ingredient_detection') {
            historyHTML += `<li><strong>${date} ${time}</strong>: Ingredient detection ${feedback.correct ? 'correct' : 'incorrect'}</li>`;
        } else {
            const ratings = Object.values(feedback.ratings).filter(r => r > 0);
            if (ratings.length > 0) {
                const avgRating = ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
                historyHTML += `<li><strong>${date} ${time}</strong>: Recipe rating ${avgRating.toFixed(1)}/5 stars</li>`;
            } else {
                historyHTML += `<li><strong>${date} ${time}</strong>: Recipe feedback submitted</li>`;
            }
        }
    });
    historyHTML += '</ul>';
    
    historyContent.innerHTML = historyHTML;
}

// Utility Functions
function showNotification(message, type = 'info') {
    console.log('Notification:', message);
    
    // Create or update notification
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '1000';
        notification.style.maxWidth = '300px';
        notification.style.padding = '12px';
        notification.style.borderRadius = '8px';
        notification.style.fontWeight = '500';
        document.body.appendChild(notification);
    }
    
    // Set notification style based on type
    notification.className = `status status--${type}`;
    notification.textContent = message;
    notification.style.display = 'block';
    
    setTimeout(() => {
        notification.style.display = 'none';
    }, 4000);
}

// Session State Management (simplified for sandbox)
function saveSessionState() {
    try {
        const stateToSave = {
            currentTab: appState.currentTab,
            detectedIngredients: appState.detectedIngredients,
            confirmedIngredients: appState.confirmedIngredients,
            preferences: appState.preferences,
            currentRecipe: appState.currentRecipe,
            feedbackHistory: appState.feedback.history
        };
        
        // Using window object for state persistence in sandbox
        window.appStateBackup = stateToSave;
        console.log('State saved');
    } catch (error) {
        console.log('State saving simulated:', error.message);
    }
}

function loadSessionState() {
    try {
        const savedState = window.appStateBackup;
        if (savedState) {
            console.log('Loading saved state...');
            
            appState.currentTab = savedState.currentTab || 'detection';
            appState.detectedIngredients = savedState.detectedIngredients || [];
            appState.confirmedIngredients = savedState.confirmedIngredients || [];
            appState.preferences = { ...appState.preferences, ...savedState.preferences };
            appState.currentRecipe = savedState.currentRecipe;
            appState.feedback.history = savedState.feedbackHistory || [];
            
            // Switch to saved tab
            if (appState.currentTab !== 'detection') {
                switchTab(appState.currentTab);
            }
            
            // Restore UI state
            if (appState.detectedIngredients.length > 0) {
                const resultsDiv = document.getElementById('detectionResults');
                if (resultsDiv) {
                    resultsDiv.classList.remove('hidden');
                    resultsDiv.style.display = 'block';
                    displayDetectedIngredients();
                }
            }
            
            if (appState.currentRecipe) {
                displayRecipe(appState.currentRecipe);
                const recipeResults = document.getElementById('recipeResults');
                const recipeDisplay = document.getElementById('recipeDisplay');
                if (recipeResults) {
                    recipeResults.classList.remove('hidden');
                    recipeResults.style.display = 'block';
                }
                if (recipeDisplay) {
                    recipeDisplay.classList.remove('hidden');
                    recipeDisplay.style.display = 'block';
                }
            }
            
            console.log('State loaded successfully');
        }
    } catch (error) {
        console.log('State loading simulated:', error.message);
    }
}