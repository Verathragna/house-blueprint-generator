import React, { useState } from 'react';

interface BlueprintData {
  floorPlan: string;
  elevation: string;
  dimensions: {
    width: number;
    length: number;
    totalArea: number;
  };
}

const DesignFixed = () => {
  const [step, setStep] = useState(1);
  const [isGenerating, setIsGenerating] = useState(false);
  const [blueprintData, setBlueprintData] = useState<BlueprintData | null>(null);
  const [errorMessage, setErrorMessage] = useState('');

  const [formData, setFormData] = useState({
    squareFootage: '',
    bedrooms: '',
    bathrooms: '',
    floors: '',
    style: '',
    lotWidth: '',
    lotLength: '',
    budget: ''
  });

  const architecturalStyles = [
    'Modern', 'Contemporary', 'Traditional', 'Colonial', 
    'Mediterranean', 'Craftsman', 'Ranch', 'Victorian'
  ];

  const budgetRanges = ['200-300k', '300-400k', '400-500k', '500k+'];
  const budgetLabels: Record<string, string> = {
    '200-300k': '$200,000 - $300,000',
    '300-400k': '$300,000 - $400,000',
    '400-500k': '$400,000 - $500,000',
    '500k+': '$500,000+'
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Basic validation
    if (!formData.squareFootage || !formData.bedrooms || !formData.bathrooms || 
        !formData.floors || !formData.style || !formData.lotWidth || 
        !formData.lotLength || !formData.budget) {
      setErrorMessage('Please fill in all required fields.');
      return;
    }

    setIsGenerating(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock blueprint data
      setBlueprintData({
        floorPlan: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0iI2Y0ZjRmNCIvPgo8dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE4IiBmaWxsPSIjMzMzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Rmxvb3IgUGxhbjwvdGV4dD4KPHN2Zz4K',
        elevation: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0iI2Y0ZjRmNCIvPgo8dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE4IiBmaWxsPSIjMzMzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+RWxldmF0aW9uPC90ZXh0Pgo8L3N2Zz4K',
        dimensions: {
          width: parseInt(formData.lotWidth),
          length: parseInt(formData.lotLength),
          totalArea: parseInt(formData.squareFootage)
        }
      });
      setStep(2);
    } catch (error) {
      setErrorMessage('Error generating blueprint. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div>
      {isGenerating && (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-40">
          <div className="bg-white p-4 rounded-md flex items-center">
            <div className="animate-spin h-5 w-5 mr-2 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            Generating blueprint...
          </div>
        </div>
      )}
      
      {errorMessage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white rounded-lg p-6 shadow-lg max-w-sm w-full">
            <p className="text-gray-800 mb-4">{errorMessage}</p>
            <button
              className="mt-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              onClick={() => setErrorMessage('')}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div className="pt-16 min-h-screen bg-gray-50">
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
          {step === 1 ? (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h1 className="text-2xl font-bold mb-6">Design Your Dream Home</h1>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                      {budgetRanges.map(range => (
                        <option key={range} value={range}>
                          {budgetLabels[range]}
                        </option>
                      ))}
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

                <div className="flex justify-end">
                  <button
                    type="submit"
                    className="bg-blue-600 text-white px-6 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={isGenerating}
                  >
                    {isGenerating ? 'Generating Blueprint...' : 'Generate Blueprint'}
                  </button>
                </div>
              </form>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Floor Plan</h3>
                    <div className="bg-gray-100 rounded-lg overflow-auto min-h-[60vh] flex items-center justify-center">
                      {blueprintData && (
                        <img
                          src={blueprintData.floorPlan}
                          alt="Floor Plan"
                          className="max-w-full max-h-[70vh] object-contain mx-auto block"
                        />
                      )}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium mb-2">Elevation</h3>
                    <div className="bg-gray-100 rounded-lg overflow-auto min-h-[60vh] flex items-center justify-center">
                      {blueprintData && (
                        <img
                          src={blueprintData.elevation}
                          alt="Elevation"
                          className="max-w-full max-h-[70vh] object-contain mx-auto block"
                        />
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex justify-between mt-6">
                  <div className="flex space-x-2">
                    <button className="flex items-center px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
                      Save
                    </button>
                    <button className="flex items-center px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
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
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DesignFixed;