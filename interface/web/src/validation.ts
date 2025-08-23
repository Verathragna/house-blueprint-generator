import { z } from 'zod';

export const architecturalStyles = [
  'Modern',
  'Contemporary',
  'Traditional',
  'Colonial',
  'Mediterranean',
  'Craftsman',
  'Ranch',
  'Victorian'
] as const;

export const specialFeatures = [
  'Garage',
  'Basement',
  'Home Office',
  'Open Floor Plan',
  'Master Suite',
  'Outdoor Kitchen',
  'Pool',
  'Solar Panels'
] as const;

export const budgetRanges = [
  '200-300k',
  '300-400k',
  '400-500k',
  '500k+'
] as const;

export const budgetLabels: Record<typeof budgetRanges[number], string> = {
  '200-300k': '$200,000 - $300,000',
  '300-400k': '$300,000 - $400,000',
  '400-500k': '$400,000 - $500,000',
  '500k+': '$500,000+'
};

export const designFormSchema = z.object({
  squareFootage: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) > 0, { message: 'Must be greater than 0' }),
  bedrooms: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) >= 0, { message: 'Must be 0 or more' }),
  bathrooms: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) >= 0, { message: 'Must be 0 or more' }),
  floors: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) >= 1 && Number(v) <= 4, {
      message: 'Must be between 1 and 4'
    }),
  style: z.enum(architecturalStyles, { required_error: 'Required' }),
  lotWidth: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) > 0, { message: 'Must be greater than 0' }),
  lotLength: z
    .string()
    .min(1, 'Required')
    .regex(/^\d+$/, 'Must be a number')
    .refine(v => Number(v) > 0, { message: 'Must be greater than 0' }),
  specialRequirements: z.array(z.enum(specialFeatures)).optional(),
  budget: z.enum(budgetRanges, { required_error: 'Required' })
});

export type DesignFormData = z.infer<typeof designFormSchema>;

