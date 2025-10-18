import React, { useState } from 'react';

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const faqs = [
    {
      question: 'What is included in a blueprint design package?',
      answer: 'Our blueprint design packages typically include floor plans, elevation drawings, electrical layouts, plumbing diagrams, and structural details. Each package also comes with material recommendations, cost estimates, and up to three revision rounds depending on the selected tier.'
    },
    {
      question: 'How long does the design process take?',
      answer: 'The typical design process takes 4-6 weeks from initial consultation to final blueprint delivery. This timeline can vary based on project complexity, revision requests, and package type. Rush services are available for an additional fee.'
    },
    {
      question: 'Do you help with permit applications?',
      answer: 'Yes, our Premium package includes permit application support. We ensure all blueprints meet local building codes and regulations, and we can assist with documentation required for permit submission.'
    },
    {
      question: 'Can I modify my design after it\'s completed?',
      answer: 'Yes, each package includes a specific number of revision rounds. Additional revisions can be purchased if needed. We recommend finalizing all major decisions during the initial design phase to avoid extra costs.'
    },
    {
      question: 'What information do I need to provide to start the design process?',
      answer: 'To begin, we need your lot dimensions, desired square footage, number of rooms, architectural style preferences, budget range, and any special requirements. Our online design tool will guide you through providing all necessary information.'
    },
    {
      question: 'Do you provide 3D visualization of the designs?',
      answer: 'Yes, our Professional and Premium packages include 3D exterior visualization. The Premium package also includes full interior visualization and virtual walkthrough capabilities.'
    },
    {
      question: 'How do you handle specific local building codes?',
      answer: 'Our team stays updated with local building codes across different regions. We customize each design to comply with your specific area\'s requirements, including setbacks, height restrictions, and zoning regulations.'
    },
    {
      question: 'What payment methods do you accept?',
      answer: 'We accept all major credit cards, bank transfers, and PayPal. Payment plans are available for larger projects, typically requiring a 50% deposit to begin work and the remaining balance upon completion.'
    },
    {
      question: 'Can you work with unusual lot shapes or sloped terrain?',
      answer: 'Yes, we specialize in creating custom designs for challenging lots. Our team has experience with irregular lot shapes, sloped terrain, and waterfront properties. These factors will be considered in the initial design phase.'
    },
    {
      question: 'Do you provide cost estimates for construction?',
      answer: 'Yes, all our packages include preliminary cost estimates based on current local construction rates. However, final costs may vary based on contractor selection, material choices, and market conditions.'
    },
    {
      question: 'What if I\'m not satisfied with the design?',
      answer: 'Customer satisfaction is our priority. If you\'re not happy with the initial design, we\'ll work with you through the included revision rounds to make necessary adjustments. We also offer a satisfaction guarantee on all our packages.'
    },
    {
      question: 'Can you incorporate eco-friendly and sustainable features?',
      answer: 'Absolutely! We can integrate various sustainable features such as solar panel preparation, energy-efficient layouts, passive heating/cooling design, and eco-friendly material recommendations. Just specify your green building goals during the design phase.'
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 animate-card">Frequently Asked Questions</h1>
          <p className="text-xl max-w-2xl mx-auto animate-card animation-delay-100">
            Find answers to common questions about our blueprint design services and process.
          </p>
        </div>
      </div>

      {/* FAQ Accordion Section */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-white rounded-lg shadow-md overflow-hidden animate-card"
              style={{ animationDelay: `${(index + 1) * 100}ms` }}
            >
              <button
                className="w-full px-6 py-4 flex justify-between items-center hover:bg-gray-50 transition-colors"
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
              >
                <span className="text-lg font-medium text-gray-900">{faq.question}</span>
                {openIndex === index ? (
                  <ChevronUp className="h-5 w-5 text-gray-500" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-500" />
                )}
              </button>
              {openIndex === index && (
                <div className="px-6 py-4 bg-gray-50">
                  <p className="text-gray-600">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Contact CTA Section */}
      <div className="bg-gray-50 py-20">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6 animate-card">Still Have Questions?</h2>
          <p className="text-gray-600 mb-8 animate-card animation-delay-100">
            Can't find the answer you're looking for? Our team is here to help.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4 animate-card animation-delay-200">
            <button className="bg-blue-600 text-white px-8 py-3 rounded-md font-medium hover:bg-blue-700 transition-colors">
              Contact Support
            </button>
            <button className="bg-gray-200 text-gray-900 px-8 py-3 rounded-md font-medium hover:bg-gray-300 transition-colors">
              Schedule Consultation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FAQ;