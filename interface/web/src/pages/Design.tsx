import React, { useState } from 'react';
import { Save, Download, RefreshCw, Loader2 } from 'lucide-react';

/**
 * @typedef {Object} DesignFormData
 * @property {string} squareFootage - Total square footage of the house
 * @property {string} bedrooms - Number of bedrooms
 * @property {string} bathrooms - Number of bathrooms
 * @property {string} floors - Number of floors
 * @property {string} style - Architectural style preference
 * @property {string} lotWidth - Width of the lot in feet
 * @property {string} lotLength - Length of the lot in feet
 * @property {string[]} specialRequirements - Array of special features requested
 * @property {string} budget - Budget range for the project
 */
interface DesignFormData {
  squareFootage: string;
  bedrooms: string;
  bathrooms: string;
  floors: string;
  style: string;
  lotWidth: string;
  lotLength: string;
  specialRequirements: string[];
  budget: string;
}

/**
 * @typedef {Object} BlueprintData
 * @property {string} floorPlan - URL to the generated floor plan image
 * @property {string} elevation - URL to the generated elevation drawing
 * @property {Object} dimensions - Calculated dimensions and measurements
 */
interface BlueprintData {
  floorPlan: string;
  elevation: string;
  dimensions: {
    width: number;
    length: number;
    totalArea: number;
  };
}

/**
 * Design page component that handles the blueprint design process.
 * 
 * State Management:
 * - step: Tracks current step in design process (1: form, 2: preview)
 * - isGenerating: Controls loading state during blueprint generation
 * - formData: Stores all user input in a centralized state object
 * - blueprintData: Stores generated blueprint information
 * 
 * @component
 * @returns {JSX.Element} Rendered Design page component
 */
const DesignPage = () => {
  // Track current step in the design process
  const [step, setStep] = useState(1);
  
  // Control loading state during blueprint generation
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Store generated blueprint data
  const [blueprintData, setBlueprintData] = useState<BlueprintData | null>(null);
  
  // Centralized state for form data
  const [formData, setFormData] = useState<DesignFormData>({
    squareFootage: '',
    bedrooms: '',
    bathrooms: '',
    floors: '',
    style: '',
    lotWidth: '',
    lotLength: '',
    specialRequirements: [],
    budget: ''
  });

  /**
   * Simulates blueprint generation based on form data.
   * In a production environment, this would call an actual blueprint generation service.
   * 
   * @param {DesignFormData} data - Form data used to generate the blueprint
   * @returns {Promise<BlueprintData>} Generated blueprint data
   */
  const generateBlueprint = async (data: DesignFormData): Promise<BlueprintData> => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 3000));

    // PLACE API CALL HERE:
    // Replace this mock implementation with your actual API call
    // Example:
    // try {
    //   const response = await fetch('YOUR_API_ENDPOINT', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify(data)
    //   });
    //   
    //   if (!response.ok) {
    //     throw new Error('Blueprint generation failed');
    //   }
    //   
    //   return await response.json();
    // } catch (error) {
    //   console.error('Error generating blueprint:', error);
    //   throw error;
    // }

    // Mock response - replace with actual API response
    return {
      floorPlan: 'https://images.unsplash.com/photo-1574706472779-9b0f28c9a240?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      elevation: 'https://images.unsplash.com/photo-1574706472779-9b0f28c9a240?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      dimensions: {
        width: parseInt(data.lotWidth),
        length: parseInt(data.lotLength),
        totalArea: parseInt(data.squareFootage)
      }
    };
  };

  const architecturalStyles = [
    'Modern',
    'Contemporary',
    'Traditional',
    'Colonial',
    'Mediterranean',
    'Craftsman',
    'Ranch',
    'Victorian'
  ];

  const specialFeatures = [
    'Garage',
    'Basement',
    'Home Office',
    'Open Floor Plan',
    'Master Suite',
    'Outdoor Kitchen',
    'Pool',
    'Solar Panels'
  ];

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleCheckboxChange = (feature: string) => {
    setFormData(prev => ({
      ...prev,
      specialRequirements: prev.specialRequirements.includes(feature)
        ? prev.specialRequirements.filter(f => f !== feature)
        : [...prev.specialRequirements, feature]
    }));
  };

  const validateForm = () => {
    const { squareFootage, bedrooms, bathrooms, floors, style, lotWidth, lotLength, budget } = formData;
    return (
      squareFootage && 
      bedrooms && 
      bathrooms && 
      floors && 
      style && 
      lotWidth && 
      lotLength && 
      budget &&
      Number(squareFootage) > 0 &&
      Number(bedrooms) > 0 &&
      Number(bathrooms) > 0 &&
      Number(floors) > 0 &&
      Number(lotWidth) > 0 &&
      Number(lotLength) > 0
    );
  };

  /**
   * Handles form submission and blueprint generation.
   * Validates input, shows loading state, and transitions to preview step.
   * 
   * Process:
   * 1. Prevent default form submission
   * 2. Validate form data
   * 3. Show loading state
   * 4. Generate blueprint
   * 5. Store blueprint data
   * 6. Transition to preview step
   * 
   * @param {React.FormEvent} e - Form submission event
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) {
      alert('Please fill in all required fields with valid values');
      return;
    }
    
    setIsGenerating(true);
    try {
      const generatedBlueprint = await generateBlueprint(formData);
      setBlueprintData(generatedBlueprint);
      setStep(2);
    } catch (error) {
      alert('Error generating blueprint. Please try again.');
      console.error('Blueprint generation error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="pt-16 min-h-screen bg-gray-50">
      {/* Progress indicator */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${step === 1 ? '50%' : '100%'}` }}
              ></div>
            </div>
            <div className="flex justify-between mt-2 text-sm text-gray-600">
              <span>Design Parameters</span>
              <span>Blueprint Preview</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Step 1: Design Parameters Form */}
        {step === 1 ? (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h1 className="text-2xl font-bold mb-6">Design Your Dream Home</h1>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Form fields for house specifications */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Total Square Footage*
                  </label>
                  <input
                    type="number"
                    name="squareFootage"
                    value={formData.squareFootage}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Bedrooms*
                  </label>
                  <input
                    type="number"
                    name="bedrooms"
                    value={formData.bedrooms}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Bathrooms*
                  </label>
                  <input
                    type="number"
                    name="bathrooms"
                    value={formData.bathrooms}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Number of Floors*
                  </label>
                  <input
                    type="number"
                    name="floors"
                    value={formData.floors}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="1"
                    max="4"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Lot Width (feet)*
                  </label>
                  <input
                    type="number"
                    name="lotWidth"
                    value={formData.lotWidth}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="20"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Lot Length (feet)*
                  </label>
                  <input
                    type="number"
                    name="lotLength"
                    value={formData.lotLength}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                    min="20"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Budget Range*
                  </label>
                  <select
                    name="budget"
                    value={formData.budget}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                  >
                    <option value="">Select a range</option>
                    <option value="200-300k">$200,000 - $300,000</option>
                    <option value="300-400k">$300,000 - $400,000</option>
                    <option value="400-500k">$400,000 - $500,000</option>
                    <option value="500k+">$500,000+</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Architectural Style*
                  </label>
                  <select
                    name="style"
                    value={formData.style}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    required
                  >
                    <option value="">Select a style</option>
                    {architecturalStyles.map(style => (
                      <option key={style} value={style}>{style}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Special requirements section */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Special Requirements
                </label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {specialFeatures.map(feature => (
                    <label key={feature} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={formData.specialRequirements.includes(feature)}
                        onChange={() => handleCheckboxChange(feature)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">{feature}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Form submission button */}
              <div className="flex justify-end">
                <button
                  type="submit"
                  className="bg-blue-600 text-white px-6 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={isGenerating}
                >
                  {isGenerating ? (
                    <span className="flex items-center">
                      <Loader2 className="animate-spin h-5 w-5 mr-2" />
                      Generating Blueprint...
                    </span>
                  ) : (
                    'Generate Blueprint'
                  )}
                </button>
              </div>
            </form>
          </div>
        ) : (
          /* Step 2: Blueprint Preview */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Blueprint preview panel */}
            <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-2">Floor Plan</h3>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 rounded-lg overflow-hidden">
                    {blueprintData && (
                      <img 
                        src={blueprintData.floorPlan} 
                        alt="Floor Plan" 
                        className="object-cover w-full h-full"
                      />
                    )}
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-medium mb-2">Elevation</h3>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 rounded-lg overflow-hidden">
                    {blueprintData && (
                      <img 
                        src={blueprintData.elevation} 
                        alt="Elevation" 
                        className="object-cover w-full h-full"
                      />
                    )}
                  </div>
                </div>
              </div>
              <div className="flex justify-between mt-6">
                <div className="flex space-x-2">
                  <button className="flex items-center px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
                    <Save className="h-5 w-5 mr-2" />
                    Save
                  </button>
                  <button className="flex items-center px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
                    <Download className="h-5 w-5 mr-2" />
                    Export
                  </button>
                </div>
                <button 
                  className="px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                  onClick={() => setStep(1)}
                >
                  Modify Design
                </button>
              </div>
            </div>

            {/* Design details sidebar */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Design Details</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-700">Dimensions</h3>
                  <p className="text-gray-600">
                    {formData.squareFootage} sq ft total
                    <br />
                    Lot: {formData.lotWidth}' × {formData.lotLength}'
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-gray-700">Layout</h3>
                  <p className="text-gray-600">
                    {formData.bedrooms} bedrooms
                    <br />
                    {formData.bathrooms} bathrooms
                    <br />
                    {formData.floors} floor(s)
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-gray-700">Style</h3>
                  <p className="text-gray-600">{formData.style}</p>
                </div>
                {formData.specialRequirements.length > 0 && (
                  <div>
                    <h3 className="font-medium text-gray-700">Special Features</h3>
                    <ul className="list-disc list-inside text-gray-600">
                      {formData.specialRequirements.map(feature => (
                        <li key={feature}>{feature}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DesignPage;