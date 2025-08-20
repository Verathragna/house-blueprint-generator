import React from 'react';
import { Check } from 'lucide-react';

const Pricing = () => {
  const plans = [
    {
      name: 'Basic',
      price: 1499,
      description: 'Perfect for small renovation projects',
      features: [
        '2D Floor Plans',
        'Basic Elevation Drawings',
        'Material Recommendations',
        'One Revision Round',
        'Digital Delivery',
        'Email Support'
      ]
    },
    {
      name: 'Professional',
      price: 2999,
      description: 'Ideal for new home construction',
      features: [
        'Everything in Basic',
        '3D Exterior Visualization',
        'Detailed Construction Drawings',
        'Three Revision Rounds',
        'Project Timeline',
        'Priority Support'
      ],
      popular: true
    },
    {
      name: 'Premium',
      price: 4999,
      description: 'Complete design and planning solution',
      features: [
        'Everything in Professional',
        'Full 3D Interior & Exterior',
        'Virtual Walkthrough',
        'Unlimited Revisions',
        'Permit Application Support',
        '24/7 Priority Support'
      ]
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 animate-card">Simple, Transparent Pricing</h1>
          <p className="text-xl max-w-2xl mx-auto animate-card animation-delay-100">
            Choose the perfect plan for your project. All plans include our commitment to quality and excellence.
          </p>
        </div>
      </div>

      {/* Pricing Cards */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <div
              key={index}
              className={`relative rounded-lg animate-card ${
                plan.popular
                  ? 'bg-white shadow-xl border-2 border-blue-600 scale-105'
                  : 'bg-white shadow-lg'
              }`}
              style={{ animationDelay: `${(index + 1) * 100}ms` }}
            >
              {plan.popular && (
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <span className="bg-blue-600 text-white px-4 py-1 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
              )}
              <div className="p-8">
                <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                <p className="text-gray-600 mb-6">{plan.description}</p>
                <div className="mb-8">
                  <span className="text-4xl font-bold">${plan.price}</span>
                  <span className="text-gray-600">/project</span>
                </div>
                <ul className="space-y-4 mb-8">
                  {plan.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start">
                      <Check className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                <button
                  className={`w-full py-3 px-6 rounded-md font-medium transition-colors ${
                    plan.popular
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  }`}
                >
                  Get Started
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* FAQ Section */}
      <div className="bg-gray-50 py-20">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-8 animate-card">Frequently Asked Questions</h2>
          <p className="text-gray-600 mb-8 animate-card animation-delay-100">
            Have questions about our pricing? Contact us for custom quotes or special requirements.
          </p>
          <button className="bg-blue-600 text-white px-8 py-3 rounded-md font-medium hover:bg-blue-700 transition-colors animate-card animation-delay-200">
            Contact Sales
          </button>
        </div>
      </div>
    </div>
  );
};

export default Pricing;